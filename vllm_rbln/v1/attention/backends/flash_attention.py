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
"""Attention layer with FlashAttention."""

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Optional

import torch
from vllm.attention.backends.abstract import (AttentionBackend, AttentionImpl,
                                              AttentionMetadata, AttentionType)
from vllm.v1.attention.backends.utils import CommonAttentionMetadata
from vllm.v1.kv_cache_interface import AttentionSpec
from vllm.v1.worker.block_table import BlockTable

if TYPE_CHECKING:
    from vllm.v1.core.sched.output import SchedulerOutput
    from vllm.v1.worker.gpu_input_batch import InputBatch
    from vllm_rbln.v1.worker.rbln_model_runner import RBLNModelRunner

import vllm_rbln.rbln_envs as envs
from vllm_rbln.logger import init_logger

logger = init_logger(__name__)


# RBLN custom op (flash attention naive prefill/decode)
@torch.library.custom_op("rbln_custom_ops::flash_attention_naive_prefill",
                         mutates_args=())
def flash_attention_naive_prefill_impl(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    kv_cache: torch.Tensor,
    mask: torch.Tensor,
    scale: torch.Tensor,
    seq_idx: torch.Tensor,
    block_table: torch.Tensor,
    slot_mapping: torch.Tensor,
) -> torch.Tensor:
    if not envs.RBLN_COMPILE_MODEL:
        # attn_weights = MM(q,kt) * scale
        # attn_weights = add(attn_weights + mask)
        # attn_weights = softmax(attn_weights)
        # MM(attn_weights, v)
        partition = kv_cache.size(-2)
        seq_len = q.size(-2)
        # s = seq_idx[0][0]
        s = seq_idx[0]
        e = s + seq_len
        # block = block_table[0]
        block = block_table[0][0]
        k_state = (kv_cache[0][block].unsqueeze(0).slice_scatter(k,
                                                                 dim=3,
                                                                 start=s,
                                                                 end=e))
        v_state = (kv_cache[1][block].unsqueeze(0).slice_scatter(v,
                                                                 dim=3,
                                                                 start=s,
                                                                 end=e))
        attn_weights = torch.matmul(q, k_state.transpose(3, 4)) * scale
        causal_mask = torch.where(mask[:, :, :, :, :partition] > 0, 0.0,
                                  -float("inf"))
        attn_weights = attn_weights + causal_mask
        attn_weights = torch.nn.functional.softmax(attn_weights, dim=-1)
        attn_output = torch.matmul(attn_weights, v_state)
        return attn_output
    else:
        return torch.empty_like(q)


@torch.library.register_fake("rbln_custom_ops::flash_attention_naive_prefill")
def _(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    kv_cache: torch.Tensor,
    mask: torch.Tensor,
    scale: torch.Tensor,
    seq_idx: torch.Tensor,
    block_table: torch.Tensor,
    slot_mapping: torch.Tensor,
) -> torch.Tensor:
    return torch.empty_like(q)


@torch.library.custom_op("rbln_custom_ops::flash_attention_naive_decode",
                         mutates_args=())
def flash_attention_naive_decode_impl(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    kv_cache: torch.Tensor,
    mask: torch.Tensor,
    scale: torch.Tensor,
    seq_idx: torch.Tensor,
    block_table: torch.Tensor,
    slot_mapping: torch.Tensor,
) -> torch.Tensor:
    if not envs.RBLN_COMPILE_MODEL:
        # NOTE - multiple decode kernel implementation is necessary
        assert q.size(0) == 1
        partition = kv_cache.size(-2)
        seq_len = q.size(-2)
        s = seq_idx[0][0]
        e = s + seq_len
        block = block_table[0]
        k_state = (kv_cache[0][block].unsqueeze(0).slice_scatter(k,
                                                                 dim=3,
                                                                 start=s,
                                                                 end=e))
        v_state = (kv_cache[1][block].unsqueeze(0).slice_scatter(v,
                                                                 dim=3,
                                                                 start=s,
                                                                 end=e))
        attn_weights = torch.matmul(q, k_state.transpose(3, 4)) * scale
        causal_mask = torch.where(mask[:, :, :, :, :partition] > 0, 0.0,
                                  -float("inf"))
        attn_weights = attn_weights + causal_mask
        attn_weights = torch.nn.functional.softmax(attn_weights, dim=-1)
        attn_output = torch.matmul(attn_weights, v_state)
        return attn_output
    else:
        return torch.empty_like(q)


@torch.library.register_fake("rbln_custom_ops::flash_attention_naive_decode")
def _(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    kv_cache: torch.Tensor,
    mask: torch.Tensor,
    scale: torch.Tensor,
    seq_idx: torch.Tensor,
    block_table: torch.Tensor,
    slot_mapping: torch.Tensor,
) -> torch.Tensor:
    return torch.empty_like(q)


# RBLN custom op (cache update)
# NYI, custom op interface is only registered for test
# inputs = {cache, state, batch, seq}
@torch.library.custom_op("rbln_custom_ops::rbln_cache_update", mutates_args=())
def rbln_cache_update_impl(cache: torch.Tensor, state: torch.Tensor,
                           slot_mapping: torch.Tensor) -> torch.Tensor:
    return torch.empty_like(cache)


@torch.library.register_fake("rbln_custom_ops::rbln_cache_update")
def _(cache: torch.Tensor, state: torch.Tensor,
      slot_mapping: torch.Tensor) -> torch.Tensor:
    return torch.empty_like(cache)


class RBLNAttentionBackend(AttentionBackend):

    @staticmethod
    def get_supported_head_sizes() -> list[int]:
        return [32, 64, 96, 128, 160, 192, 224, 256]

    @staticmethod
    def get_name() -> str:
        return "RBLN_ATTN"

    @staticmethod
    def get_impl_cls() -> type["RBLNFlashAttentionImpl"]:
        return RBLNFlashAttentionImpl

    @staticmethod
    def get_metadata_cls() -> type["AttentionMetadata"]:
        return RBLNFlashAttentionMetadata

    @staticmethod
    def get_builder_cls() -> type["RBLNFlashAttentionMetadataBuilder"]:
        return RBLNFlashAttentionMetadataBuilder

    @staticmethod
    def get_kv_cache_shape(
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_size: int,
    ) -> tuple[int, ...]:
        """kv cache shape
        # B - num_blocks == num_partitions
        # S - block_size == partition_size
        # H - num_kv_heads
        # G - num_heads / num_kv_heads = 32/8 = 4
        # D - head_size
        # L - q_len
        list of kv cache = [num_layer][kv=2]
        kv_cache_shape= [B, H, 1, S, D]
        query_shape   = [1, H, G, L, D]
        """
        return (2, num_blocks, num_kv_heads, 1, block_size, head_size)

    @staticmethod
    def swap_blocks(
        src_kv_cache: torch.Tensor,
        dst_kv_cache: torch.Tensor,
        src_to_dst: dict[int, int],
    ) -> None:
        raise RuntimeError("swap_blocks is not used for the RBLN backend.")

    @staticmethod
    def copy_blocks(
        kv_caches: list[torch.Tensor],
        src_to_dists: dict[int, list[int]],
    ) -> None:
        raise RuntimeError("swap_blocks is not used for the RBLN backend.")


@dataclass
class RBLNFlashAttentionMetadata:
    num_actual_tokens: int  # Number of tokens excluding padding.
    max_query_len: int
    query_start_loc: torch.Tensor
    max_seq_len: int
    seq_lens: torch.Tensor
    block_table: torch.Tensor
    slot_mapping: torch.Tensor

    # For cascade attention.
    use_cascade: Optional[bool]
    common_prefix_len: Optional[int]
    cu_prefix_query_lens: Optional[torch.Tensor]
    prefix_kv_lens: Optional[torch.Tensor]
    suffix_kv_lens: Optional[torch.Tensor]

    # Optional aot scheduling
    scheduler_metadata: Optional[torch.Tensor] = None
    prefix_scheduler_metadata: Optional[torch.Tensor] = None

    # for local attention
    local_attn_metadata: Optional[object] = None

    # For RBLN Attention
    attn_masks: Optional[torch.Tensor] = None
    kv_caches: Optional[list[torch.Tensor]] = None


class RBLNFlashAttentionMetadataBuilder:

    def __init__(
        self,
        runner: "RBLNModelRunner",
        kv_cache_spec: AttentionSpec,
        block_table: BlockTable,
    ):
        model_config = runner.model_config

        self.runner = runner
        self.input_batch = runner.input_batch
        self.num_heads_q = model_config.get_num_attention_heads(
            runner.parallel_config)
        self.num_heads_kv = model_config.get_num_kv_heads(
            runner.parallel_config)
        self.headdim = model_config.get_head_size()
        self.block_size = kv_cache_spec.block_size
        self.kv_cache_spec = kv_cache_spec
        self.block_table = block_table

        self.chunked_prefill = (runner.scheduler_config.chunked_prefill_enabled
                                or runner.cache_config.enable_prefix_caching)
        self.chunked_prefill_size = (
            runner.scheduler_config.max_num_batched_tokens)

    def reorder_batch(self, input_batch: "InputBatch",
                      scheduler_output: "SchedulerOutput") -> bool:
        return False

    def build(
        self,
        num_reqs: int,
        num_actual_tokens: int,
        max_query_len: int,
        common_prefix_len: int,
        common_attn_metadata: CommonAttentionMetadata,
    ) -> RBLNFlashAttentionMetadata:
        query_max_seq_len = int(self.runner.seq_lens_np[:num_reqs].max())
        query_start_loc = common_attn_metadata.query_start_loc
        seq_lens = common_attn_metadata.seq_lens
        block_table = self.block_table
        block_table_tensor = block_table.get_device_tensor()[:num_reqs]

        block_table.slot_mapping[:num_actual_tokens].copy_(
            block_table.slot_mapping_cpu[:num_actual_tokens],
            non_blocking=True)
        block_table.slot_mapping[num_actual_tokens:].fill_(-1)

        slot_mapping = block_table.slot_mapping[:num_actual_tokens]

        cu_prefix_query_lens = None
        prefix_kv_lens = None
        suffix_kv_lens = None
        prefix_scheduler_metadata = None

        seq_idx = self.runner.positions_cpu[:num_reqs].view(-1, 1)

        # The length of the partition equals the block size.
        partition_len = self.block_size
        # no. of block(HW constraint) determines max sequence length.
        # max_model_len(Model constraint) determines max sequence length.
        # One of them is selected for max_seq_len.
        block_length = self.runner.cache_config.num_gpu_blocks * partition_len
        max_seq_len = min(self.runner.model_config.max_model_len, block_length)

        num_partition = max_seq_len // partition_len
        cs = seq_idx.repeat(1, num_partition)
        pidx = torch.arange(num_partition, dtype=torch.int32)
        # RBLN - seq_lens tensor dtype SHOULD be int16
        dyn_size_for_partitions = torch.clamp(cs - pidx * partition_len, 0,
                                              partition_len).to(torch.int16)
        seq_lens_tensor = dyn_size_for_partitions

        is_prefills = (self.input_batch.num_computed_tokens_cpu
                       < self.input_batch.num_prompt_tokens)
        # The prefill and decode cannot be mixed.
        assert len(is_prefills) > 0 and all(
            is_prefill == is_prefills[0]
            for is_prefill in is_prefills[:num_reqs])
        if is_prefills[0]:
            prefill_chunk_size = (self.chunked_prefill_size
                                  if self.chunked_prefill else 1 <<
                                  (math.ceil(math.log2(query_max_seq_len))))
            chunked_attention_mask = torch.zeros(
                1,
                1,
                1,
                prefill_chunk_size,
                max_seq_len,
                dtype=torch.float32,
            )
            causal_mask = 1 - torch.triu(
                torch.ones(1, 1, prefill_chunk_size, prefill_chunk_size),
                diagonal=1,
            )
            step = seq_idx[0]
            if step >= prefill_chunk_size:
                chunked_attention_mask[:, :, :, :, :step] = 1
            chunked_attention_mask[:, :, :, :, step:step +
                                   prefill_chunk_size] = causal_mask
            attn_masks = chunked_attention_mask
        else:
            # batch padding
            batch_size = num_reqs
            batch_padding_size = self.runner.max_num_seqs - batch_size
            seq_lens_tensor = torch.cat([
                seq_lens_tensor,
                torch.full(
                    (batch_padding_size, seq_lens_tensor.shape[-1]),
                    0,
                ),
            ])
            block_table_tensor = torch.cat([
                block_table_tensor,
                torch.full(
                    (batch_padding_size, block_table_tensor.shape[-1]),
                    block_table_tensor.numel() - 1,
                ),
            ])
            decode_attention_mask = torch.zeros(
                self.runner.max_num_seqs,
                1,
                1,
                1,
                max_seq_len,
                dtype=torch.float32,
            )
            for batch_index, batch_step in enumerate(seq_lens):
                decode_attention_mask[batch_index, :, :, :, :batch_step +
                                      1] = 1
            attn_masks = decode_attention_mask

        attn_metadata = RBLNFlashAttentionMetadata(
            num_actual_tokens=num_actual_tokens,
            max_query_len=max_query_len,
            query_start_loc=query_start_loc,
            max_seq_len=query_max_seq_len,
            seq_lens=seq_lens_tensor,
            # TODO(jiwoo.park) assume single batch, single partition
            # The block table should be fixed for multiple partitions.
            # block_table=block_table_tensor,
            block_table=block_table_tensor,
            slot_mapping=slot_mapping,
            use_cascade=False,
            common_prefix_len=common_prefix_len,
            scheduler_metadata=None,
            cu_prefix_query_lens=cu_prefix_query_lens,
            prefix_kv_lens=prefix_kv_lens,
            suffix_kv_lens=suffix_kv_lens,
            local_attn_metadata=None,
            prefix_scheduler_metadata=prefix_scheduler_metadata,
            attn_masks=attn_masks,
            kv_caches=None,
        )

        logger.info("RBLNAttentionMetadata = %s", attn_metadata)
        logger.info("\tslot_mapping size = %s", slot_mapping.size())
        logger.info("\tblock_table size = %s", block_table_tensor.size())
        logger.info("\tattn_masks size = %s", attn_masks.size())
        logger.info("\tattn_masks = %s", attn_masks[:, :, :, :, :32])
        logger.info("\tseq_lens_tensor size= %s", seq_lens_tensor.size())
        return attn_metadata

    def use_cascade_attention(self, *args, **kwargs) -> bool:
        return False


class RBLNFlashAttentionImpl(AttentionImpl[RBLNFlashAttentionMetadata]):

    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float,
        num_kv_heads: int,
        alibi_slopes: Optional[list[float]],
        sliding_window: Optional[int],
        kv_cache_dtype: str,
        blocksparse_params: Optional[dict[str, Any]] = None,
        logits_soft_cap: Optional[float] = None,
        attn_type: str = AttentionType.DECODER,
        kv_sharing_target_layer_name: Optional[str] = None,
        use_irope: bool = False,
    ) -> None:
        if kv_sharing_target_layer_name is not None:
            raise NotImplementedError("KV sharing is not supported in RBLN.")
        if blocksparse_params is not None:
            raise ValueError("RBLN Attention Backend does not "
                             "support block-sparse attention.")
        if logits_soft_cap is not None:
            logger.warning_once(
                "RBLN Attention Backend does not support logits soft cap. "
                "Outputs may be slightly off.")
            logits_soft_cap = None
        if use_irope:
            logger.warning_once(
                "Using irope in RBLN Attention Backend is not supported yet, "
                "it will fall back to global attention for long context.")
            self.use_irope = False

        self.num_heads = num_heads
        self.head_size = head_size
        self.scale = torch.tensor(scale)
        self.num_kv_heads = num_kv_heads
        if alibi_slopes is not None:
            alibi_slopes = torch.tensor(alibi_slopes, dtype=torch.float32)
        self.alibi_slopes = alibi_slopes
        self.sliding_window = sliding_window
        self.kv_cache_dtype = kv_cache_dtype
        if logits_soft_cap is None:
            # In flash-attn, setting logits_soft_cap as 0 means no soft cap.
            logits_soft_cap = 0
        self.logits_soft_cap = logits_soft_cap
        self.kv_sharing_target_layer_name = kv_sharing_target_layer_name

        assert self.num_heads % self.num_kv_heads == 0
        self.num_queries_per_kv = self.num_heads // self.num_kv_heads
        self.need_mask = (self.alibi_slopes is not None
                          or self.sliding_window is not None)

        supported_head_sizes = RBLNAttentionBackend.get_supported_head_sizes()
        if head_size not in supported_head_sizes:
            raise ValueError(
                f"Head size {head_size} is not supported by PagedAttention. "
                f"Supported head sizes are: {supported_head_sizes}.")
        if kv_cache_dtype != "auto":
            raise NotImplementedError(
                "Torch SDPA backend does not support FP8 KV cache. "
                "Please use xFormers backend instead.")
        self.attn_type = attn_type

    def forward(
        self,
        layer: torch.nn.Module,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: RBLNFlashAttentionMetadata,
        output: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass with xFormers and PagedAttention.

        Args:
            query:  shape = [num_tokens, num_heads * head_size]
            key:    shape = [num_tokens, num_kv_heads * head_size]
            value:  shape = [num_tokens, num_kv_heads * head_size]
            kv_cache shape= [2, num_blocks,
                                block_size * num_kv_heads * head_size]

        Shape that we expect:
            kv_cache  = [2][num_blocks, num_kv_heads, 1, block_size, head_size]
            key       = [1, num_kv_heads, 1, block_size, head_size]
            query     = [1, num_kv_heads, 4, query_len, head_size]
            key_t     = [1, num_kv_heads, 1, head_size, block_size]
        Returns:
            attn_out  = [num_tokens, num_heads * head_size]

            hidden_size = num_heads * head_size
        """
        # B - num_blocks == num_partitions
        # S - block_size == partition_size
        # H - num_kv_heads
        # G - num_heads / num_kv_heads = 4
        # D - head_size
        # L - query length
        # C - max_seq_len
        # NB- num batch

        # 1. query reshape for custom operation
        # query = [b_size(batch), q_len(query len), num_heads * head_size]
        b_size, q_len, _ = query.size()
        query = query.view(b_size, q_len, self.num_heads,
                           self.head_size).transpose(1, 2)
        query = query.view(
            b_size,
            self.num_kv_heads,
            self.num_queries_per_kv,
            q_len,
            self.head_size,
        )
        key = key.view(b_size, q_len, self.num_kv_heads,
                       self.head_size).transpose(1, 2)
        key = key.view(b_size, self.num_kv_heads, 1, q_len, self.head_size)
        value = value.view(b_size, q_len, self.num_kv_heads,
                           self.head_size).transpose(1, 2)
        value = value.view(b_size, self.num_kv_heads, 1, q_len, self.head_size)

        # NOTE - for cache update,
        # slot mapping will be necessary from sequence index
        # slot_mapping = [block_number, block_offset]

        # flash_attention_naive extended to have cache update
        # cache update is included into flash attention
        # but not within partition loop
        # input = {q, k, v, kv_cache, mask, scalar_scale,
        # seq_lens, block_table, slot_mapping}
        # output = {attn_output}
        # q, k, v = [batch,H,G,L,D]
        # key/value cache = [B,H,1,S,D]
        # mask  = [1,1,1,L,C]
        # o = [batch,H,G,L,D]

        # build attention mask within [0, 1]
        # - attention mask SHOULD be causal mask based on query length
        # - attention mask is used for masked softmax not actual value
        # if there is not positional embedding,
        # it can be merged into attention mask
        # attn_masks = _make_alibi_bias(alibi_slopes, dtype, seq_lens)
        # seq_lens_tensor (1, num_partition = 128k / k = 128)
        # ex) tensor[partition0 = 1024, partition1 = 10,
        # partition2 = 0, partition3 = 0] for len=1034
        # block_tables tensor (1, num_blocks = 256)
        # ex) tensor[block0 : 0, block1 : 100,
        #  block2: 10, block3: 5, ...]
        # attn_output = [batch,H,4,L,D]
        assert kv_cache is not None

        # kv cache update
        if not envs.RBLN_COMPILE_MODEL:
            # s = attn_metadata.seq_lens.to(torch.int16)[0][0]
            s = attn_metadata.seq_lens.to(torch.int16)[0]
            e = s + q_len
            # block = attn_metadata.block_table.to(torch.int16)[0]
            block = attn_metadata.block_table.to(torch.int16)[0][0]
            k_state = (kv_cache[0][block].unsqueeze(0).slice_scatter(key,
                                                                     dim=3,
                                                                     start=s,
                                                                     end=e))
            v_state = (kv_cache[1][block].unsqueeze(0).slice_scatter(value,
                                                                     dim=3,
                                                                     start=s,
                                                                     end=e))
            kv_cache[0][block] = k_state.squeeze(0)
            kv_cache[1][block] = v_state.squeeze(0)

        if q_len == 1:
            attn_output = (
                torch.ops.rbln_custom_ops.flash_attention_naive_decode(
                    query,
                    key,
                    value,
                    kv_cache,
                    attn_metadata.attn_masks,
                    self.scale,
                    attn_metadata.seq_lens.to(torch.int16),
                    attn_metadata.block_table.to(torch.int16),
                    self.scale,
                ))
        else:
            # actually non-flash paged attention DOES NOT use slot_mapping
            attn_output = (
                torch.ops.rbln_custom_ops.flash_attention_naive_prefill(
                    query,
                    key,
                    value,
                    kv_cache,
                    attn_metadata.attn_masks,
                    self.scale,
                    attn_metadata.seq_lens.to(torch.int16),
                    attn_metadata.block_table.to(torch.int16),
                    self.scale,
                ))

        # 2. attention output reshape for attention backend return
        # attn_output = [batch,H*4,L,D] -> [batch,L,H*4,D] -> [batch,L,H*4*D]
        if not envs.RBLN_COMPILE_MODEL:
            attn_output = attn_output.reshape(b_size, self.num_heads, q_len,
                                              self.head_size).transpose(1, 2)
            attn_output = attn_output.reshape(b_size, q_len,
                                              self.num_heads * self.head_size)
        else:
            attn_output = attn_output.view(b_size, self.num_heads, q_len,
                                           self.head_size).transpose(1, 2)
            attn_output = attn_output.view(b_size, q_len,
                                           self.num_heads * self.head_size)
        # attn_output = [batch,L,H*4*D]
        return attn_output
