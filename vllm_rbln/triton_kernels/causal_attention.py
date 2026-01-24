# Copyright 2025 Rebellions Inc. All rights reserved.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at:

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Triton kernels for Causal Attention"""

import torch
from rebel import triton
from rebel.triton import language as tl
from rebel.triton.language.extra.rbln import libdevice as rblib
from torch.library import register_fake, triton_op


@triton.jit
def flash_causal_attention_naive_prefill(
    query,
    key,
    value,
    kv_cache,
    output,
    qk_scale,
    seq_idx,
    block_table,
    block_size,
    NUM_HEAD: tl.constexpr,
    NUM_GROUP: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    QUERY_LEN: tl.constexpr,
    NUM_BATCH: tl.constexpr,
    PARTITION_SIZE: tl.constexpr,
    MAX_SEQ_LEN: tl.constexpr,
    NUM_BLOCK: tl.constexpr,
    DIM_BLOCK_TABLE: tl.constexpr,
):
    tl.static_assert(HEAD_DIM % 64 == 0)
    tl.static_assert(MAX_SEQ_LEN % PARTITION_SIZE == 0)
    NUM_PARTITION: tl.constexpr = MAX_SEQ_LEN // PARTITION_SIZE
    DYNAMIC_AXIS: tl.constexpr = 4
    tl.static_assert(NUM_BATCH == 1)

    for batch_id in tl.static_range(0, NUM_BATCH, 1):
        query_ptr = tl.make_block_ptr(
            base=query,
            shape=(NUM_BATCH, NUM_HEAD, NUM_GROUP, QUERY_LEN, HEAD_DIM),
            strides=(
                NUM_GROUP * NUM_HEAD * QUERY_LEN * HEAD_DIM,
                NUM_GROUP * QUERY_LEN * HEAD_DIM,
                QUERY_LEN * HEAD_DIM,
                HEAD_DIM,
                1,
            ),
            offsets=(batch_id, 0, 0, 0, 0),
            block_shape=(1, NUM_HEAD, NUM_GROUP, QUERY_LEN, HEAD_DIM),
            order=(4, 3, 2, 1, 0),
        )
        output_ptr = tl.make_block_ptr(
            base=output,
            shape=(NUM_BATCH, NUM_HEAD, NUM_GROUP, QUERY_LEN, HEAD_DIM),
            strides=(
                NUM_GROUP * NUM_HEAD * QUERY_LEN * HEAD_DIM,
                NUM_GROUP * QUERY_LEN * HEAD_DIM,
                QUERY_LEN * HEAD_DIM,
                HEAD_DIM,
                1,
            ),
            offsets=(batch_id, 0, 0, 0, 0),
            block_shape=(1, NUM_HEAD, NUM_GROUP, QUERY_LEN, HEAD_DIM),
            order=(4, 3, 2, 1, 0),
        )

        key_ptr = tl.make_block_ptr(
            base=key,
            shape=(NUM_BATCH, NUM_HEAD, 1, QUERY_LEN, HEAD_DIM),
            strides=(
                1 * NUM_HEAD * QUERY_LEN * HEAD_DIM,
                1 * QUERY_LEN * HEAD_DIM,
                QUERY_LEN * HEAD_DIM,
                HEAD_DIM,
                1,
            ),
            offsets=(batch_id, 0, 0, 0, 0),
            block_shape=(1, NUM_HEAD, 1, QUERY_LEN, HEAD_DIM),
            order=(4, 3, 2, 1, 0),
        )
        value_ptr = tl.make_block_ptr(
            base=value,
            shape=(NUM_BATCH, NUM_HEAD, 1, QUERY_LEN, HEAD_DIM),
            strides=(
                1 * NUM_HEAD * QUERY_LEN * HEAD_DIM,
                1 * QUERY_LEN * HEAD_DIM,
                QUERY_LEN * HEAD_DIM,
                HEAD_DIM,
                1,
            ),
            offsets=(batch_id, 0, 0, 0, 0),
            block_shape=(1, NUM_HEAD, 1, QUERY_LEN, HEAD_DIM),
            order=(4, 3, 2, 1, 0),
        )
        q = tl.load(query_ptr)
        k_state = tl.load(key_ptr)
        v_state = tl.load(value_ptr)
        k_state = tl.reshape(k_state, (1, 1, NUM_HEAD, 1, QUERY_LEN, HEAD_DIM))
        v_state = tl.reshape(v_state, (1, 1, NUM_HEAD, 1, QUERY_LEN, HEAD_DIM))

        attn_out_prev = tl.zeros(
            [1, NUM_HEAD, NUM_GROUP, QUERY_LEN, HEAD_DIM], dtype=tl.float32
        )
        row_sum_prev = tl.zeros(
            [1, NUM_HEAD, NUM_GROUP, QUERY_LEN, 1], dtype=tl.float32
        )
        row_max_prev = tl.zeros(
            [1, NUM_HEAD, NUM_GROUP, QUERY_LEN, 1], dtype=tl.float32
        )

        for partition_id in tl.static_range(0, NUM_PARTITION, 1):
            block_table_ptr = tl.make_block_ptr(
                base=block_table,
                shape=(NUM_PARTITION,),
                strides=(1,),
                offsets=(partition_id,),
                block_shape=(1,),
                order=(0,),
            )
            tl.static_assert(
                len(block_table_ptr.type.element_ty.shape) == DIM_BLOCK_TABLE
            )
            seq_idx_ptr = tl.make_block_ptr(
                base=seq_idx,
                shape=(NUM_BATCH, NUM_PARTITION),
                strides=(NUM_PARTITION, 1),
                offsets=(batch_id, partition_id),
                block_shape=(1, 1),
                order=(1, 0),
            )

            block_number = rblib.to_dynamic_index(block_table_ptr)
            block_offset = rblib.to_dynamic_index(seq_idx_ptr, PARTITION_SIZE)
            block_number = block_number.cast(tl.int32)
            block_offset = block_offset.cast(tl.int32)

            if rblib.partition_skip(block_offset) == False:  # noqa: E712
                k_cache_ptr = tl.make_block_ptr(
                    base=kv_cache,
                    shape=(2, NUM_BLOCK, NUM_HEAD, 1, PARTITION_SIZE, HEAD_DIM),
                    strides=(
                        NUM_BLOCK * 1 * NUM_HEAD * PARTITION_SIZE * HEAD_DIM,
                        1 * NUM_HEAD * PARTITION_SIZE * HEAD_DIM,
                        1 * PARTITION_SIZE * HEAD_DIM,
                        PARTITION_SIZE * HEAD_DIM,
                        HEAD_DIM,
                        1,
                    ),
                    offsets=(0, block_number, 0, 0, 0, 0),
                    block_shape=(1, 1, NUM_HEAD, 1, PARTITION_SIZE, HEAD_DIM),
                    order=(5, 4, 3, 2, 1, 0),
                )
                k = rblib.dynamic_load(k_cache_ptr, DYNAMIC_AXIS, block_offset)
                k_insert = rblib.insert(k, k_state, DYNAMIC_AXIS, block_offset)
                rblib.dynamic_store(
                    k_cache_ptr,
                    k_insert,
                    DYNAMIC_AXIS,
                    block_offset + QUERY_LEN,
                )

                k_insert = tl.reshape(
                    k_insert, (1, NUM_HEAD, 1, PARTITION_SIZE, HEAD_DIM)
                )
                k = tl.permute(k_insert, (0, 1, 2, 4, 3))
                k = tl.broadcast_to(
                    k, (1, NUM_HEAD, NUM_GROUP, HEAD_DIM, PARTITION_SIZE)
                )
                qk = tl.dot(q, k)

                qk_scaled = qk * qk_scale

                if partition_id > 0:
                    row_max_global, row_exp_normalize, row_sum_cur = (
                        rblib.dynamic_flash_attn_tile(
                            qk_scaled, block_offset, row_max_prev
                        )
                    )
                else:
                    row_max_global, row_exp_normalize, row_sum_cur = (
                        rblib.dynamic_flash_attn_tile(qk_scaled, block_offset)
                    )

                v_cache_ptr = tl.make_block_ptr(
                    base=kv_cache,
                    shape=(2, NUM_BLOCK, NUM_HEAD, 1, PARTITION_SIZE, HEAD_DIM),
                    strides=(
                        NUM_BLOCK * 1 * NUM_HEAD * PARTITION_SIZE * HEAD_DIM,
                        1 * NUM_HEAD * PARTITION_SIZE * HEAD_DIM,
                        1 * PARTITION_SIZE * HEAD_DIM,
                        PARTITION_SIZE * HEAD_DIM,
                        HEAD_DIM,
                        1,
                    ),
                    offsets=(1, block_number, 0, 0, 0, 0),
                    block_shape=(1, 1, NUM_HEAD, 1, PARTITION_SIZE, HEAD_DIM),
                    order=(5, 4, 3, 2, 1, 0),
                )

                v = rblib.dynamic_load(v_cache_ptr, DYNAMIC_AXIS, block_offset)
                v_insert = rblib.insert(v, v_state, DYNAMIC_AXIS, block_offset)
                rblib.dynamic_store(
                    v_cache_ptr,
                    v_insert,
                    DYNAMIC_AXIS,
                    block_offset + QUERY_LEN,
                )

                v_insert = tl.reshape(
                    v_insert, (1, NUM_HEAD, 1, PARTITION_SIZE, HEAD_DIM)
                )
                v = tl.broadcast_to(
                    v_insert, (1, NUM_HEAD, NUM_GROUP, PARTITION_SIZE, HEAD_DIM)
                )
                attn_out_cur = tl.dot(row_exp_normalize, v)
                if partition_id > 0:
                    row_sum_prev, attn_out_prev = rblib.flash_attn_recompute(
                        row_max_prev,
                        row_max_global,
                        row_sum_prev,
                        row_sum_cur,
                        attn_out_prev,
                        attn_out_cur,
                    )
                else:
                    row_sum_prev = row_sum_cur
                    attn_out_prev = attn_out_cur
                row_max_prev = row_max_global

        attn_out = attn_out_prev / row_sum_prev
        tl.store(output_ptr, attn_out)


@triton.jit
def flash_causal_attention_naive_decode(
    query,
    key,
    value,
    kv_cache,
    output,
    qk_scale,
    seq_idx,
    block_table,
    block_size,
    NUM_HEAD: tl.constexpr,
    NUM_GROUP: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    QUERY_LEN: tl.constexpr,
    NUM_BATCH: tl.constexpr,
    PARTITION_SIZE: tl.constexpr,
    MAX_SEQ_LEN: tl.constexpr,
    NUM_BLOCK: tl.constexpr,
    DIM_BLOCK_TABLE: tl.constexpr,
):
    tl.static_assert(HEAD_DIM % 64 == 0)
    tl.static_assert(QUERY_LEN == 1)
    tl.static_assert(MAX_SEQ_LEN % PARTITION_SIZE == 0)
    NUM_PARTITION: tl.constexpr = MAX_SEQ_LEN // PARTITION_SIZE
    DYNAMIC_AXIS: tl.constexpr = 4
    tl.static_assert(NUM_BATCH >= 1)

    for batch_id in tl.static_range(0, NUM_BATCH, 1):
        query_ptr = tl.make_block_ptr(
            base=query,
            shape=(NUM_BATCH, NUM_HEAD, NUM_GROUP, QUERY_LEN, HEAD_DIM),
            strides=(
                NUM_GROUP * NUM_HEAD * QUERY_LEN * HEAD_DIM,
                NUM_GROUP * QUERY_LEN * HEAD_DIM,
                QUERY_LEN * HEAD_DIM,
                HEAD_DIM,
                1,
            ),
            offsets=(batch_id, 0, 0, 0, 0),
            block_shape=(1, NUM_HEAD, NUM_GROUP, QUERY_LEN, HEAD_DIM),
            order=(4, 3, 2, 1, 0),
        )
        output_ptr = tl.make_block_ptr(
            base=output,
            shape=(NUM_BATCH, NUM_HEAD, NUM_GROUP, QUERY_LEN, HEAD_DIM),
            strides=(
                NUM_GROUP * NUM_HEAD * QUERY_LEN * HEAD_DIM,
                NUM_GROUP * QUERY_LEN * HEAD_DIM,
                QUERY_LEN * HEAD_DIM,
                HEAD_DIM,
                1,
            ),
            offsets=(batch_id, 0, 0, 0, 0),
            block_shape=(1, NUM_HEAD, NUM_GROUP, QUERY_LEN, HEAD_DIM),
            order=(4, 3, 2, 1, 0),
        )
        key_ptr = tl.make_block_ptr(
            base=key,
            shape=(NUM_BATCH, NUM_HEAD, 1, QUERY_LEN, HEAD_DIM),
            strides=(
                1 * NUM_HEAD * QUERY_LEN * HEAD_DIM,
                1 * QUERY_LEN * HEAD_DIM,
                QUERY_LEN * HEAD_DIM,
                HEAD_DIM,
                1,
            ),
            offsets=(batch_id, 0, 0, 0, 0),
            block_shape=(1, NUM_HEAD, 1, QUERY_LEN, HEAD_DIM),
            order=(4, 3, 2, 1, 0),
        )
        value_ptr = tl.make_block_ptr(
            base=value,
            shape=(NUM_BATCH, NUM_HEAD, 1, QUERY_LEN, HEAD_DIM),
            strides=(
                1 * NUM_HEAD * QUERY_LEN * HEAD_DIM,
                1 * QUERY_LEN * HEAD_DIM,
                QUERY_LEN * HEAD_DIM,
                HEAD_DIM,
                1,
            ),
            offsets=(batch_id, 0, 0, 0, 0),
            block_shape=(1, NUM_HEAD, 1, QUERY_LEN, HEAD_DIM),
            order=(4, 3, 2, 1, 0),
        )
        q = tl.load(query_ptr)
        k_state = tl.load(key_ptr)
        v_state = tl.load(value_ptr)
        k_state = tl.reshape(k_state, (1, 1, NUM_HEAD, 1, QUERY_LEN, HEAD_DIM))
        v_state = tl.reshape(v_state, (1, 1, NUM_HEAD, 1, QUERY_LEN, HEAD_DIM))

        attn_out_prev = tl.zeros(
            [1, NUM_HEAD, NUM_GROUP, QUERY_LEN, HEAD_DIM], dtype=tl.float32
        )
        row_sum_prev = tl.zeros(
            [1, NUM_HEAD, NUM_GROUP, QUERY_LEN, 1], dtype=tl.float32
        )
        row_max_prev = tl.zeros(
            [1, NUM_HEAD, NUM_GROUP, QUERY_LEN, 1], dtype=tl.float32
        )

        for partition_id in tl.static_range(0, NUM_PARTITION, 1):
            block_table_ptr = tl.make_block_ptr(
                base=block_table,
                shape=(NUM_BATCH, NUM_PARTITION),
                strides=(NUM_PARTITION, 1),
                offsets=(batch_id, partition_id),
                block_shape=(1, 1),
                order=(1, 0),
            )
            tl.static_assert(
                len(block_table_ptr.type.element_ty.shape) == DIM_BLOCK_TABLE
            )
            seq_idx_ptr = tl.make_block_ptr(
                base=seq_idx,
                shape=(NUM_BATCH, NUM_PARTITION),
                strides=(NUM_PARTITION, 1),
                offsets=(batch_id, partition_id),
                block_shape=(1, 1),
                order=(1, 0),
            )

            block_number = rblib.to_dynamic_index(block_table_ptr)
            block_offset = rblib.to_dynamic_index(seq_idx_ptr, PARTITION_SIZE)
            block_number = block_number.cast(tl.int32)
            block_offset = block_offset.cast(tl.int32)

            if rblib.partition_skip(block_offset) == False:  # noqa: E712
                k_cache_ptr = tl.make_block_ptr(
                    base=kv_cache,
                    shape=(2, NUM_BLOCK, NUM_HEAD, 1, PARTITION_SIZE, HEAD_DIM),
                    strides=(
                        NUM_BLOCK * 1 * NUM_HEAD * PARTITION_SIZE * HEAD_DIM,
                        1 * NUM_HEAD * PARTITION_SIZE * HEAD_DIM,
                        1 * PARTITION_SIZE * HEAD_DIM,
                        PARTITION_SIZE * HEAD_DIM,
                        HEAD_DIM,
                        1,
                    ),
                    offsets=(0, block_number, 0, 0, 0, 0),
                    block_shape=(1, 1, NUM_HEAD, 1, PARTITION_SIZE, HEAD_DIM),
                    order=(5, 4, 3, 2, 1, 0),
                )
                k = rblib.dynamic_load(k_cache_ptr, DYNAMIC_AXIS, block_offset)
                k_insert = rblib.insert(k, k_state, DYNAMIC_AXIS, block_offset)
                rblib.dynamic_store(
                    k_cache_ptr,
                    k_insert,
                    DYNAMIC_AXIS,
                    block_offset + QUERY_LEN,
                )

                k_insert = tl.reshape(
                    k_insert, (1, NUM_HEAD, 1, PARTITION_SIZE, HEAD_DIM)
                )
                k = tl.permute(k_insert, (0, 1, 2, 4, 3))
                k = tl.broadcast_to(
                    k, (1, NUM_HEAD, NUM_GROUP, HEAD_DIM, PARTITION_SIZE)
                )
                qk = tl.dot(q, k)

                qk_scaled = qk * qk_scale

                if partition_id > 0:
                    row_max_global, row_exp_normalize, row_sum_cur = (
                        rblib.dynamic_flash_attn_tile(
                            qk_scaled, block_offset, row_max_prev
                        )
                    )
                else:
                    row_max_global, row_exp_normalize, row_sum_cur = (
                        rblib.dynamic_flash_attn_tile(qk_scaled, block_offset)
                    )

                v_cache_ptr = tl.make_block_ptr(
                    base=kv_cache,
                    shape=(2, NUM_BLOCK, NUM_HEAD, 1, PARTITION_SIZE, HEAD_DIM),
                    strides=(
                        NUM_BLOCK * 1 * NUM_HEAD * PARTITION_SIZE * HEAD_DIM,
                        1 * NUM_HEAD * PARTITION_SIZE * HEAD_DIM,
                        1 * PARTITION_SIZE * HEAD_DIM,
                        PARTITION_SIZE * HEAD_DIM,
                        HEAD_DIM,
                        1,
                    ),
                    offsets=(1, block_number, 0, 0, 0, 0),
                    block_shape=(1, 1, NUM_HEAD, 1, PARTITION_SIZE, HEAD_DIM),
                    order=(5, 4, 3, 2, 1, 0),
                )

                v = rblib.dynamic_load(v_cache_ptr, DYNAMIC_AXIS, block_offset)
                v_insert = rblib.insert(v, v_state, DYNAMIC_AXIS, block_offset)
                rblib.dynamic_store(
                    v_cache_ptr,
                    v_insert,
                    DYNAMIC_AXIS,
                    block_offset + QUERY_LEN,
                )

                v_insert = tl.reshape(
                    v_insert, (1, NUM_HEAD, 1, PARTITION_SIZE, HEAD_DIM)
                )
                v = tl.broadcast_to(
                    v_insert, (1, NUM_HEAD, NUM_GROUP, PARTITION_SIZE, HEAD_DIM)
                )
                attn_out_cur = tl.dot(row_exp_normalize, v)
                if partition_id > 0:
                    row_sum_prev, attn_out_prev = rblib.flash_attn_recompute(
                        row_max_prev,
                        row_max_global,
                        row_sum_prev,
                        row_sum_cur,
                        attn_out_prev,
                        attn_out_cur,
                    )
                else:
                    row_sum_prev = row_sum_cur
                    attn_out_prev = attn_out_cur
                row_max_prev = row_max_global

        attn_out = attn_out_prev / row_sum_prev
        tl.store(output_ptr, attn_out)


def warmup(func, *args):
    kernel = func.warmup(*args, grid=(1,), host_layout="1:2:3")
    rblib.write_rtosa(kernel, args)

    return kernel


@triton_op("rbln_triton_ops::flash_causal_attention_naive_prefill", mutates_args=())
def _(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    kv_cache: torch.Tensor,
    qk_scale: torch.Tensor,
    seq_idx: torch.Tensor,
    block_table: torch.Tensor,
    dummy0: torch.Tensor,
) -> torch.Tensor:
    original_dtype = query.dtype

    query = query.to(torch.float32)
    key = key.to(torch.float32)
    value = value.to(torch.float32)
    kv_cache = kv_cache.to(torch.float32)
    qk_scale = qk_scale.to(torch.float32)

    output = torch.empty_like(query)

    NUM_HEAD = query.shape[1]
    NUM_GROUP = query.shape[2]
    HEAD_DIM = query.shape[-1]
    QUERY_LEN = query.shape[-2]
    PARTITION_SIZE = kv_cache.shape[-2]
    MAX_SEQ_LEN = PARTITION_SIZE * seq_idx.shape[1]
    NUM_BLOCK = kv_cache.shape[1]
    NUM_BATCH = query.shape[0]
    DIM_BLOCK_TABLE = block_table.dim()

    params = [
        query,
        key,
        value,
        kv_cache,
        output,
        qk_scale,
        seq_idx,
        block_table,
        qk_scale,
        NUM_HEAD,
        NUM_GROUP,
        HEAD_DIM,
        QUERY_LEN,
        NUM_BATCH,
        PARTITION_SIZE,
        MAX_SEQ_LEN,
        NUM_BLOCK,
        DIM_BLOCK_TABLE,
    ]

    warmup(flash_causal_attention_naive_prefill, *params)

    return output.to(original_dtype)


@register_fake("rbln_triton_ops::flash_causal_attention_naive_prefill")
def _(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    kv_cache: torch.Tensor,
    qk_scale: torch.Tensor,
    seq_idx: torch.Tensor,
    block_table: torch.Tensor,
    dummy0: torch.Tensor,
) -> torch.Tensor:
    return torch.empty_like(query)


@triton_op("rbln_triton_ops::flash_causal_attention_naive_decode", mutates_args=())
def _(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    kv_cache: torch.Tensor,
    qk_scale: torch.Tensor,
    seq_idx: torch.Tensor,
    block_table: torch.Tensor,
    dummy0: torch.Tensor,
) -> torch.Tensor:
    original_dtype = query.dtype

    query = query.to(torch.float32)
    key = key.to(torch.float32)
    value = value.to(torch.float32)
    kv_cache = kv_cache.to(torch.float32)
    qk_scale = qk_scale.to(torch.float32)

    output = torch.empty_like(query)

    NUM_HEAD = query.shape[1]
    NUM_GROUP = query.shape[2]
    HEAD_DIM = query.shape[-1]
    QUERY_LEN = query.shape[-2]
    PARTITION_SIZE = kv_cache.shape[-2]
    MAX_SEQ_LEN = PARTITION_SIZE * seq_idx.shape[1]
    NUM_BLOCK = kv_cache.shape[1]
    NUM_BATCH = query.shape[0]
    DIM_BLOCK_TABLE = block_table.dim()

    params = [
        query,
        key,
        value,
        kv_cache,
        output,
        qk_scale,
        seq_idx,
        block_table,
        qk_scale,
        NUM_HEAD,
        NUM_GROUP,
        HEAD_DIM,
        QUERY_LEN,
        NUM_BATCH,
        PARTITION_SIZE,
        MAX_SEQ_LEN,
        NUM_BLOCK,
        DIM_BLOCK_TABLE,
    ]

    warmup(flash_causal_attention_naive_decode, *params)

    return output.to(original_dtype)


@register_fake("rbln_triton_ops::flash_causal_attention_naive_decode")
def _(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    kv_cache: torch.Tensor,
    qk_scale: torch.Tensor,
    seq_idx: torch.Tensor,
    block_table: torch.Tensor,
    dummy0: torch.Tensor,
) -> torch.Tensor:
    return torch.empty_like(query)
