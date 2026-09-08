# Copyright 2025 Rebellions Inc. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at:
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch

from vllm_rbln import envs

from . import triton_flash_causal_attention  # noqa: F401


def flash_causal_attention_naive_prefill(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    kv_cache: torch.Tensor,
    scale: torch.Tensor,
    seq_idx: torch.Tensor,
    block_tables: torch.Tensor,
    sinks: torch.Tensor | None = None,
    k_quantize_scale: torch.Tensor | None = None,
    v_quantize_scale: torch.Tensor | None = None,
    cache_dtype: torch.dtype | None = None,
) -> torch.Tensor:
    if envs.VLLM_RBLN_COMPILE_MODEL:
        if envs.VLLM_RBLN_USE_CUSTOM_KERNEL:
            return torch.ops.rbln_triton_ops.flash_causal_attention_naive_prefill(
                q,
                k,
                v,
                kv_cache,
                scale,
                seq_idx,
                block_tables,
                scale,  # dummy
            )
        else:
            return torch.ops.rbln_custom_ops.flash_causal_attention_naive_prefill(
                q,
                k,
                v,
                kv_cache,
                scale,
                seq_idx,
                block_tables,
                scale,  # dummy,
                sinks,
                k_quantize_scale,
                v_quantize_scale,
                cache_dtype,
            )

    return _eager_flash_causal_attention(
        q, k, v, kv_cache, scale, seq_idx, block_tables, sinks, cache_dtype
    )


def flash_causal_attention_naive_decode(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    kv_cache: torch.Tensor,
    scale: torch.Tensor,
    seq_idx: torch.Tensor,
    block_tables: torch.Tensor,
    sinks: torch.Tensor | None = None,
    k_quantize_scale: torch.Tensor | None = None,
    v_quantize_scale: torch.Tensor | None = None,
    cache_dtype: torch.dtype | None = None,
) -> torch.Tensor:
    if envs.VLLM_RBLN_COMPILE_MODEL:
        if envs.VLLM_RBLN_USE_CUSTOM_KERNEL:
            return torch.ops.rbln_triton_ops.flash_causal_attention_naive_decode(
                q,
                k,
                v,
                kv_cache,
                scale,
                seq_idx,
                block_tables,
                scale,  # dummy,
            )
        else:
            return torch.ops.rbln_custom_ops.flash_causal_attention_naive_decode(
                q,
                k,
                v,
                kv_cache,
                scale,
                seq_idx,
                block_tables,
                scale,  # dummy,
                sinks,
                k_quantize_scale,
                v_quantize_scale,
                cache_dtype,
            )

    return _eager_flash_causal_attention(
        q, k, v, kv_cache, scale, seq_idx, block_tables, sinks, cache_dtype
    )


# ---------------------------------------------------------------------------
# Eager (VLLM_RBLN_COMPILE_MODEL=0) reference path
#
# Pure-torch implementation of the naive paged causal attention, restored for
# the eager path that the compiled-op registration (deferred to rebel-compiler)
# no longer provides. `seq_idx` is the raw cache position, shape [B, 1]; the
# block table is [P] for prefill and [B, P] for decode; fp8 KV caches are not
# supported here.
# ---------------------------------------------------------------------------


def _eager_write_kv(kv_cache, k, v, block_ids, cache_start_pos, seq_len, b):
    partition_size = kv_cache.size(-2)
    for p in range(block_ids.shape[0]):
        block_idx = int(block_ids[p].item())
        partition_start = p * partition_size
        partition_end = partition_start + partition_size
        write_start = max(cache_start_pos, partition_start)
        write_end = min(cache_start_pos + seq_len, partition_end)
        if write_start >= write_end:
            continue
        n = write_end - write_start
        off_p = write_start - partition_start
        off_i = write_start - cache_start_pos
        kv_cache[0, block_idx, :, :, off_p : off_p + n, :] = k[
            b, :, :, off_i : off_i + n, :
        ]
        kv_cache[1, block_idx, :, :, off_p : off_p + n, :] = v[
            b, :, :, off_i : off_i + n, :
        ]


def _eager_gather_kv(kv_cache, block_ids, total_seq_len, like):
    partition_size = kv_cache.size(-2)
    n_kv_heads, head_dim = kv_cache.size(2), kv_cache.size(-1)
    k_g = torch.zeros(
        1, n_kv_heads, 1, total_seq_len, head_dim, dtype=like.dtype, device=like.device
    )
    v_g = torch.zeros_like(k_g)
    pos = 0
    for p in range(block_ids.shape[0]):
        block_idx = int(block_ids[p].item())
        tokens = min(total_seq_len - p * partition_size, partition_size)
        if tokens <= 0:
            break
        k_g[:, :, :, pos : pos + tokens, :] = kv_cache[0, block_idx, :, :, :tokens, :]
        v_g[:, :, :, pos : pos + tokens, :] = kv_cache[1, block_idx, :, :, :tokens, :]
        pos += tokens
    return k_g, v_g


def _eager_attend(q_b, k_g, v_g, scale, cache_start_pos, sinks):
    _, n_kv_heads, n_groups, seq_len, _ = q_b.shape
    total_seq_len = k_g.size(-2)
    attn = torch.matmul(q_b, k_g.transpose(3, 4)) * scale
    q_pos = torch.arange(cache_start_pos, cache_start_pos + seq_len, device=q_b.device)
    k_pos = torch.arange(total_seq_len, device=q_b.device)
    mask = q_pos.unsqueeze(1) >= k_pos.unsqueeze(0)
    attn = attn + torch.where(mask, 0.0, float("-inf")).to(attn.dtype)[None, None, None]
    if sinks is not None:
        sink_len = sinks.size(-1)
        sinks_e = sinks.view(n_kv_heads, n_groups, 1, sink_len).expand(
            1, n_kv_heads, n_groups, seq_len, sink_len
        )
        combined = torch.cat([attn, sinks_e], dim=-1)
        combined = combined - combined.max(dim=-1, keepdim=True).values
        attn = torch.nn.functional.softmax(combined, dim=-1)[..., :-sink_len]
    else:
        attn = torch.nn.functional.softmax(attn, dim=-1)
    return torch.matmul(attn, v_g)


def _eager_flash_causal_attention(
    q, k, v, kv_cache, scale, seq_idx, block_tables, sinks, cache_dtype
):
    if cache_dtype is not None:
        raise NotImplementedError(
            "eager naive attention does not support fp8 KV caches"
        )
    batch_size, _, _, seq_len, _ = q.shape
    if block_tables.dim() == 1:  # prefill: a single sequence, block table [P]
        block_tables = block_tables.unsqueeze(0)
    outputs = []
    for b in range(batch_size):
        cache_start_pos = int(seq_idx[b, 0].item())
        block_ids = block_tables[b]
        _eager_write_kv(kv_cache, k, v, block_ids, cache_start_pos, seq_len, b)
        total_seq_len = cache_start_pos + seq_len
        k_g, v_g = _eager_gather_kv(kv_cache, block_ids, total_seq_len, k)
        outputs.append(
            _eager_attend(q[b : b + 1], k_g, v_g, scale, cache_start_pos, sinks)
        )
    return torch.cat(outputs, dim=0)
