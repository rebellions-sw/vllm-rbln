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
"""Triton kernels for Sliding Window Attention"""

import torch
from rebel import triton
from rebel.triton import language as tl
from rebel.triton.language.extra.rbln import libdevice as rblib
from torch.library import register_fake, triton_op


@triton.jit
def sliding_window_attention_naive_prefill(
    query,
    key,
    value,
    kv_cache,
    output,
    cache_seq_len,
    cache_offset,
    qk_scale,
    block_table,
    dummy,
    NUM_HEAD: tl.constexpr,
    NUM_GROUP: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    QUERY_LEN: tl.constexpr,
    WINDOW_SIZE: tl.constexpr,
    NUM_BATCH: tl.constexpr,
    DIM_BLOCK_TABLE: tl.constexpr,
):
    tl.static_assert(HEAD_DIM % 64 == 0)
    tl.static_assert(NUM_BATCH == 1)
    NUM_PARTITION: tl.constexpr = 1
    WINDOW_AXIS: tl.constexpr = 4

    for batch_id in tl.static_range(0, NUM_BATCH, 1):
        cache_seq_len_ptr = tl.make_block_ptr(
            base=cache_seq_len,
            shape=(NUM_BATCH, WINDOW_SIZE // WINDOW_SIZE),
            strides=(WINDOW_SIZE // WINDOW_SIZE, 1),
            offsets=(batch_id, 0),
            block_shape=(1, 1),
            order=(1, 0),
        )
        cache_start = rblib.to_dynamic_index(cache_seq_len_ptr)

        cache_offset_ptr = tl.make_block_ptr(
            base=cache_offset,
            shape=(NUM_BATCH, WINDOW_SIZE // WINDOW_SIZE),
            strides=(WINDOW_SIZE // WINDOW_SIZE, 1),
            offsets=(batch_id, 0),
            block_shape=(1, 1),
            order=(1, 0),
        )
        cache_end = rblib.to_dynamic_index(cache_offset_ptr)

        block_table_ptr = tl.make_block_ptr(
            base=block_table,
            shape=(NUM_PARTITION,),
            strides=(1,),
            offsets=(batch_id,),
            block_shape=(1,),
            order=(0,),
        )
        tl.static_assert(len(block_table_ptr.type.element_ty.shape) == DIM_BLOCK_TABLE)

        block_number = rblib.to_dynamic_index(block_table_ptr)

        k_block_ptr = tl.make_block_ptr(
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
        v_block_ptr = tl.make_block_ptr(
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
        k_cache_ptr = tl.make_block_ptr(
            base=kv_cache,
            shape=(2, NUM_BATCH, NUM_HEAD, 1, WINDOW_SIZE, HEAD_DIM),
            strides=(
                NUM_BATCH * NUM_HEAD * WINDOW_SIZE * HEAD_DIM,
                NUM_HEAD * WINDOW_SIZE * HEAD_DIM,
                WINDOW_SIZE * HEAD_DIM,
                WINDOW_SIZE * HEAD_DIM,
                HEAD_DIM,
                1,
            ),
            offsets=(0, block_number, 0, 0, 0, 0),
            block_shape=(1, 1, NUM_HEAD, 1, WINDOW_SIZE, HEAD_DIM),
            order=(5, 4, 3, 2, 1, 0),
        )
        v_cache_ptr = tl.make_block_ptr(
            base=kv_cache,
            shape=(2, NUM_BATCH, NUM_HEAD, 1, WINDOW_SIZE, HEAD_DIM),
            strides=(
                NUM_BATCH * NUM_HEAD * WINDOW_SIZE * HEAD_DIM,
                NUM_HEAD * WINDOW_SIZE * HEAD_DIM,
                WINDOW_SIZE * HEAD_DIM,
                WINDOW_SIZE * HEAD_DIM,
                HEAD_DIM,
                1,
            ),
            offsets=(1, block_number, 0, 0, 0, 0),
            block_shape=(1, 1, NUM_HEAD, 1, WINDOW_SIZE, HEAD_DIM),
            order=(5, 4, 3, 2, 1, 0),
        )

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
        q = tl.load(query_ptr)

        k_state = tl.load(k_block_ptr)
        k_state = tl.reshape(k_state, (1, 1, NUM_HEAD, 1, QUERY_LEN, HEAD_DIM))
        k = rblib.dynamic_load(k_cache_ptr)
        k_insert = rblib.window_insert(k, k_state, WINDOW_AXIS, cache_start)
        k_slice = rblib.window_slice(k_insert, WINDOW_SIZE, WINDOW_AXIS, cache_end)
        rblib.dynamic_store(k_cache_ptr, k_slice)

        k_insert = tl.reshape(
            k_insert, (1, NUM_HEAD, 1, WINDOW_SIZE + QUERY_LEN, HEAD_DIM)
        )
        k_tmp = tl.permute(k_insert, (0, 1, 2, 4, 3))
        k = tl.broadcast_to(
            k_tmp, (1, NUM_HEAD, NUM_GROUP, HEAD_DIM, WINDOW_SIZE + QUERY_LEN)
        )
        qk = tl.dot(q, k)
        qk_scaled = qk * qk_scale
        window_qk_scaled = rblib.window_softmax(qk_scaled, cache_start, WINDOW_SIZE)

        v_state = tl.load(v_block_ptr)
        v_state = tl.reshape(v_state, (1, 1, NUM_HEAD, 1, QUERY_LEN, HEAD_DIM))
        v = rblib.dynamic_load(v_cache_ptr)
        v_insert = rblib.window_insert(v, v_state, WINDOW_AXIS, cache_start)
        v_slice = rblib.window_slice(v_insert, WINDOW_SIZE, WINDOW_AXIS, cache_end)
        rblib.dynamic_store(v_cache_ptr, v_slice)

        v_insert = tl.reshape(
            v_insert, (1, NUM_HEAD, 1, WINDOW_SIZE + QUERY_LEN, HEAD_DIM)
        )
        v = tl.broadcast_to(
            v_insert,
            (1, NUM_HEAD, NUM_GROUP, WINDOW_SIZE + QUERY_LEN, HEAD_DIM),
        )
        attn_out = tl.dot(window_qk_scaled, v)
        tl.store(output_ptr, attn_out)


@triton.jit
def sliding_window_attention_naive_decode(
    query,
    key,
    value,
    kv_cache,
    output,
    cache_seq_len,
    cache_offset,
    qk_scale,
    block_table,
    dummy,
    NUM_HEAD: tl.constexpr,
    NUM_GROUP: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    QUERY_LEN: tl.constexpr,
    WINDOW_SIZE: tl.constexpr,
    NUM_BATCH: tl.constexpr,
    DIM_BLOCK_TABLE: tl.constexpr,
):
    tl.static_assert(HEAD_DIM % 64 == 0)
    tl.static_assert(QUERY_LEN == 1)
    tl.static_assert(NUM_BATCH >= 1)
    NUM_PARTITION: tl.constexpr = 1
    WINDOW_AXIS: tl.constexpr = 4
    PAD_SIZE: tl.constexpr = 63

    for batch_id in tl.static_range(0, NUM_BATCH, 1):
        block_table_ptr = tl.make_block_ptr(
            base=block_table,
            shape=(
                NUM_BATCH,
                NUM_PARTITION,
            ),
            strides=(NUM_PARTITION, 1),
            offsets=(batch_id, 0),
            block_shape=(1, 1),
            order=(1, 0),
        )
        tl.static_assert(len(block_table_ptr.type.element_ty.shape) == DIM_BLOCK_TABLE)
        cache_seq_len_ptr = tl.make_block_ptr(
            base=cache_seq_len,
            shape=(NUM_BATCH, WINDOW_SIZE // WINDOW_SIZE),
            strides=(WINDOW_SIZE // WINDOW_SIZE, 1),
            offsets=(batch_id, 0),
            block_shape=(1, 1),
            order=(1, 0),
        )
        cache_offset_ptr = tl.make_block_ptr(
            base=cache_offset,
            shape=(NUM_BATCH, WINDOW_SIZE // WINDOW_SIZE),
            strides=(WINDOW_SIZE // WINDOW_SIZE, 1),
            offsets=(batch_id, 0),
            block_shape=(1, 1),
            order=(1, 0),
        )

        cache_start = rblib.to_dynamic_index(cache_seq_len_ptr)
        cache_end = rblib.to_dynamic_index(cache_offset_ptr)
        block_number = rblib.to_dynamic_index(block_table_ptr)

        k_block_ptr = tl.make_block_ptr(
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
        v_block_ptr = tl.make_block_ptr(
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
        k_cache_ptr = tl.make_block_ptr(
            base=kv_cache,
            shape=(2, NUM_BATCH, NUM_HEAD, 1, WINDOW_SIZE, HEAD_DIM),
            strides=(
                NUM_BATCH * NUM_HEAD * WINDOW_SIZE * HEAD_DIM,
                NUM_HEAD * WINDOW_SIZE * HEAD_DIM,
                WINDOW_SIZE * HEAD_DIM,
                WINDOW_SIZE * HEAD_DIM,
                HEAD_DIM,
                1,
            ),
            offsets=(0, block_number, 0, 0, 0, 0),
            block_shape=(1, 1, NUM_HEAD, 1, WINDOW_SIZE, HEAD_DIM),
            order=(5, 4, 3, 2, 1, 0),
        )
        v_cache_ptr = tl.make_block_ptr(
            base=kv_cache,
            shape=(2, NUM_BATCH, NUM_HEAD, 1, WINDOW_SIZE, HEAD_DIM),
            strides=(
                NUM_BATCH * NUM_HEAD * WINDOW_SIZE * HEAD_DIM,
                NUM_HEAD * WINDOW_SIZE * HEAD_DIM,
                WINDOW_SIZE * HEAD_DIM,
                WINDOW_SIZE * HEAD_DIM,
                HEAD_DIM,
                1,
            ),
            offsets=(1, block_number, 0, 0, 0, 0),
            block_shape=(1, 1, NUM_HEAD, 1, WINDOW_SIZE, HEAD_DIM),
            order=(5, 4, 3, 2, 1, 0),
        )

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
        q = tl.load(query_ptr)

        k_state = tl.load(k_block_ptr)
        k_state = tl.reshape(k_state, (1, 1, NUM_HEAD, 1, QUERY_LEN, HEAD_DIM))
        k = rblib.dynamic_load(k_cache_ptr)
        k_insert = rblib.window_insert(k, k_state, WINDOW_AXIS, cache_start)
        k_slice = rblib.window_slice(k_insert, WINDOW_SIZE, WINDOW_AXIS, cache_end)
        rblib.dynamic_store(k_cache_ptr, k_slice)

        k_insert = rblib.nn_pad(k_insert, 0.0, WINDOW_AXIS, (0, PAD_SIZE), "constant")
        k_insert = tl.reshape(
            k_insert,
            (1, NUM_HEAD, 1, WINDOW_SIZE + QUERY_LEN + PAD_SIZE, HEAD_DIM),
        )
        k_insert = tl.permute(k_insert, (0, 1, 2, 4, 3))
        k = tl.broadcast_to(
            k_insert,
            (
                1,
                NUM_HEAD,
                NUM_GROUP,
                HEAD_DIM,
                WINDOW_SIZE + QUERY_LEN + PAD_SIZE,
            ),
        )
        qk = tl.dot(q, k)
        qk_scaled = qk * qk_scale
        window_qk_scaled = rblib.window_softmax(qk_scaled, cache_start, WINDOW_SIZE)

        v_state = tl.load(v_block_ptr)
        v_state = tl.reshape(v_state, (1, 1, NUM_HEAD, 1, QUERY_LEN, HEAD_DIM))
        v = rblib.dynamic_load(v_cache_ptr)
        v_insert = rblib.window_insert(v, v_state, WINDOW_AXIS, cache_start)
        v_slice = rblib.window_slice(v_insert, WINDOW_SIZE, WINDOW_AXIS, cache_end)
        rblib.dynamic_store(v_cache_ptr, v_slice)

        v_insert = rblib.nn_pad(
            v_insert, 0.0, (WINDOW_AXIS), ((0, PAD_SIZE)), "constant"
        )
        v_insert = tl.reshape(
            v_insert,
            (1, NUM_HEAD, 1, WINDOW_SIZE + QUERY_LEN + PAD_SIZE, HEAD_DIM),
        )
        v = tl.broadcast_to(
            v_insert,
            (
                1,
                NUM_HEAD,
                NUM_GROUP,
                WINDOW_SIZE + QUERY_LEN + PAD_SIZE,
                HEAD_DIM,
            ),
        )
        attn_out = tl.dot(window_qk_scaled, v)
        tl.store(output_ptr, attn_out)


def warmup(func, *args):
    kernel = func.warmup(*args, grid=(1,), host_layout="1:2:3")
    rblib.write_rtosa(kernel, args)

    return kernel


@triton_op("rbln_triton_ops::sliding_window_attention_naive_prefill", mutates_args=())
def _(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    kv_cache: torch.Tensor,
    cache_seq_len: torch.Tensor,
    cache_offset: torch.Tensor,
    qk_scale: torch.Tensor,
    block_table: torch.Tensor,
    dummy: torch.Tensor,
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
    WINDOW_SIZE = kv_cache.shape[-2]
    NUM_BATCH = query.shape[0]
    DIM_BLOCK_TABLE = block_table.dim()

    params = [
        query,
        key,
        value,
        kv_cache,
        output,
        cache_seq_len,
        cache_offset,
        qk_scale,
        block_table,
        qk_scale,
        NUM_HEAD,
        NUM_GROUP,
        HEAD_DIM,
        QUERY_LEN,
        WINDOW_SIZE,
        NUM_BATCH,
        DIM_BLOCK_TABLE,
    ]

    warmup(sliding_window_attention_naive_prefill, *params)

    return output.to(original_dtype)


@register_fake("rbln_triton_ops::sliding_window_attention_naive_prefill")
def _(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    kv_cache: torch.Tensor,
    cache_seq_len: torch.Tensor,
    cache_offset: torch.Tensor,
    qk_scale: torch.Tensor,
    block_table: torch.Tensor,
    dummy: torch.Tensor,
) -> torch.Tensor:
    return torch.empty_like(query)


@triton_op("rbln_triton_ops::sliding_window_attention_naive_decode", mutates_args=())
def _(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    kv_cache: torch.Tensor,
    cache_seq_len: torch.Tensor,
    cache_offset: torch.Tensor,
    qk_scale: torch.Tensor,
    block_table: torch.Tensor,
    dummy: torch.Tensor,
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
    WINDOW_SIZE = kv_cache.shape[-2]
    NUM_BATCH = query.shape[0]
    DIM_BLOCK_TABLE = block_table.dim()

    params = [
        query,
        key,
        value,
        kv_cache,
        output,
        cache_seq_len,
        cache_offset,
        qk_scale,
        block_table,
        qk_scale,
        NUM_HEAD,
        NUM_GROUP,
        HEAD_DIM,
        QUERY_LEN,
        WINDOW_SIZE,
        NUM_BATCH,
        DIM_BLOCK_TABLE,
    ]

    warmup(sliding_window_attention_naive_decode, *params)

    return output.to(original_dtype)


@register_fake("rbln_triton_ops::sliding_window_attention_naive_decode")
def _(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    kv_cache: torch.Tensor,
    cache_seq_len: torch.Tensor,
    cache_offset: torch.Tensor,
    qk_scale: torch.Tensor,
    block_table: torch.Tensor,
    dummy: torch.Tensor,
) -> torch.Tensor:
    return torch.empty_like(query)
