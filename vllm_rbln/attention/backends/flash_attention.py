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

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Type

import torch
from vllm.attention.backends.abstract import (AttentionBackend, AttentionImpl,
                                              AttentionLayer,
                                              AttentionMetadata,
                                              AttentionMetadataBuilder,
                                              AttentionType)
from vllm.attention.backends.utils import CommonAttentionState
from vllm.attention.ops.paged_attn import PagedAttentionMetadata

import vllm_rbln.rbln_envs as envs
from vllm_rbln.logger import init_logger
from vllm_rbln.worker.model_runner import ModelInputForRebelBuilder

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
    block_tables: torch.Tensor,
    slot_mapping: torch.Tensor,
) -> torch.Tensor:
    if not envs.RBLN_COMPILE_MODEL:
        # attn_weights = MM(q,kt) * scale
        # attn_weights = add(attn_weights + mask)
        # attn_weights = softmax(attn_weights)
        # MM(attn_weights, v)
        partition = kv_cache.size(-2)
        seq_len = q.size(-2)
        s = seq_idx[0][0]
        e = s + seq_len
        block = block_tables[0]
        k_state = kv_cache[0][block].unsqueeze(0).slice_scatter(k,
                                                                dim=3,
                                                                start=s,
                                                                end=e)
        v_state = kv_cache[1][block].unsqueeze(0).slice_scatter(v,
                                                                dim=3,
                                                                start=s,
                                                                end=e)
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
    block_tables: torch.Tensor,
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
    block_tables: torch.Tensor,
    slot_mapping: torch.Tensor,
) -> torch.Tensor:
    if not envs.RBLN_COMPILE_MODEL:
        # NOTE - multiple decode kernel implementation is necessary
        assert q.size(0) == 1
        partition = kv_cache.size(-2)
        seq_len = q.size(-2)
        s = seq_idx[0][0]
        e = s + seq_len
        block = block_tables[0]
        k_state = kv_cache[0][block].unsqueeze(0).slice_scatter(k,
                                                                dim=3,
                                                                start=s,
                                                                end=e)
        v_state = kv_cache[1][block].unsqueeze(0).slice_scatter(v,
                                                                dim=3,
                                                                start=s,
                                                                end=e)
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
    block_tables: torch.Tensor,
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
    def get_supported_head_sizes() -> List[int]:
        return [32, 64, 96, 128, 160, 192, 224, 256]

    @staticmethod
    def get_name() -> str:
        return "RBLN_ATTN"

    @staticmethod
    def get_impl_cls() -> Type["RBLNAttentionImpl"]:
        return RBLNAttentionImpl

    @staticmethod
    def get_metadata_cls() -> Type["AttentionMetadata"]:
        return RBLNAttentionMetadata

    @staticmethod
    def get_builder_cls() -> Type["RBLNAttentionMetadataBuilder"]:
        return RBLNAttentionMetadataBuilder

    @staticmethod
    def get_state_cls() -> Type["CommonAttentionState"]:
        return CommonAttentionState

    @staticmethod
    def get_kv_cache_shape(
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_size: int,
    ) -> Tuple[int, ...]:
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
        # for partition skip, we need dummy block slot.
        no_dummy_slots = 1
        return (2, num_blocks + no_dummy_slots, num_kv_heads, 1, block_size,
                head_size)

    @staticmethod
    def swap_blocks(
        src_kv_cache: torch.Tensor,
        dst_kv_cache: torch.Tensor,
        src_to_dst: Dict[int, int],
    ) -> None:
        raise RuntimeError("swap_blocks is not used for the RBLN backend.")

    @staticmethod
    def copy_blocks(
        kv_caches: List[torch.Tensor],
        src_to_dists: Dict[int, List[int]],
    ) -> None:
        raise RuntimeError("swap_blocks is not used for the RBLN backend.")


@dataclass
class RBLNAttentionMetadata(AttentionMetadata, PagedAttentionMetadata):
    attn_masks: Optional[torch.Tensor]
    kv_caches: Optional[List[torch.Tensor]]


class RBLNAttentionMetadataBuilder(
        AttentionMetadataBuilder[RBLNAttentionMetadata]):

    def __init__(self, input_builder: ModelInputForRebelBuilder) -> None:
        self.chunked_prefill = input_builder.chunked_prefill
        self.chunked_prefill_size = input_builder.chunked_prefill_size
        self.input_builder = input_builder

        self.partition_len = input_builder.block_size

    def prepare(self):
        self.input_data = self.input_builder.input_data

    def build(
        self,
        seq_lens: List[int],
        query_lens: List[int],
        input_block_ids: torch.Tensor,
        batch_size: int,
    ) -> RBLNAttentionMetadata:
        input_data = self.input_data
        # slot_mapping is treated as constant buffer -> not aligned issue
        slot_mapping = torch.tensor(input_data.slot_mapping,
                                    dtype=torch.int32,
                                    device="cpu")
        # temporarily, following tensors is used for
        # RBLN flash attention based on paging
        # - seq_lens_tensor[1][num_partitions] = dynamic size for partitions
        # - block_tables[num_partitions] = block index for partitions
        steps = [[input_positions[0]]
                 for input_positions in input_data.input_positions]
        seq_idx = torch.tensor(steps, dtype=torch.int32)
        partition_len = self.partition_len
        # no. of block(HW constraint) determines max sequence length.
        # max_model_len(Model constraint) determines max sequence length.
        # One of them is selected for max_seq_len.
        block_length = self.input_builder.runner.cache_config.num_gpu_blocks * \
                                            partition_len
        max_seq_len = min(self.input_builder.max_model_len, block_length)
        num_partition = max_seq_len // partition_len

        batch_size = 1 if input_data.num_prefills else len(steps)
        cs = seq_idx.repeat(1, num_partition)
        pidx = torch.arange(num_partition, dtype=torch.int32)
        # RBLN - seq_lens tensor dtype SHOULD be int16
        dyn_size_for_partitions = torch.clamp(cs - pidx * partition_len, 0,
                                              partition_len).to(torch.int32)
        seq_lens_tensor = dyn_size_for_partitions

        # RBLN - block_tables tensor dtype SHOULD be int16
        block_tables = input_block_ids.to(torch.int32)

        # For multi-modal models
        placeholder_index_maps = None
        if len(input_data.multi_modal_inputs_list) != 0:
            placeholder_index_maps = {
                modality: placeholder_map.index_map()
                for modality, placeholder_map in
                input_data.multi_modal_placeholder_maps.items()
            }

        # RBLN attention mask
        # prefill attention mask vs decode attention mask
        if input_data.num_prefills:
            step = steps[0][0]
            assert input_data.num_prefills == 1
            prefill_chunk_size = (
                self.chunked_prefill_size if self.chunked_prefill else 1 <<
                (math.ceil(math.log2(input_data.seq_lens[0]))))
            chunked_attention_mask = torch.zeros(1,
                                                 1,
                                                 1,
                                                 prefill_chunk_size,
                                                 max_seq_len,
                                                 dtype=torch.float32)
            causal_mask = 1 - torch.triu(torch.ones(1, 1, prefill_chunk_size,
                                                    prefill_chunk_size),
                                         diagonal=1)
            if step >= prefill_chunk_size:
                chunked_attention_mask[:, :, :, :, :step] = 1
            chunked_attention_mask[:, :, :, :, step:step +
                                   prefill_chunk_size] = causal_mask
            attn_masks = chunked_attention_mask
        else:
            decode_attention_mask = torch.zeros(batch_size,
                                                1,
                                                1,
                                                1,
                                                max_seq_len,
                                                dtype=torch.float32)
            for batch_index, batch_step in enumerate(steps):
                decode_attention_mask[batch_index, :, :, :, :batch_step[0] +
                                      1] = 1
            attn_masks = decode_attention_mask

        assert attn_masks.dim() == 5
        attn_metadata = RBLNAttentionMetadata(
            num_prefills=input_data.num_prefills,
            num_prefill_tokens=input_data.num_prefill_tokens,
            num_decode_tokens=input_data.num_decode_tokens,
            slot_mapping=slot_mapping,
            multi_modal_placeholder_index_maps=placeholder_index_maps,
            enable_kv_scales_calculation=False,
            seq_lens_tensor=seq_lens_tensor,
            max_decode_seq_len=max_seq_len,
            block_tables=block_tables,
            attn_masks=attn_masks,
            kv_caches=None,
        )
        logger.info("RBLNAttentionMetadata = %s", attn_metadata)
        logger.info("\tslot_mapping size = %s", slot_mapping.size())
        logger.info("\tblock_tables size = %s", block_tables.size())
        logger.info("\tattn_masks size = %s", attn_masks.size())
        logger.info("\tattn_masks = %s", attn_masks[:, :, :, :, :32])
        logger.info("\tseq_lens_tensor size= %s", seq_lens_tensor.size())
        return attn_metadata


class RBLNAttentionImpl(AttentionImpl[RBLNAttentionMetadata]):

    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float,
        num_kv_heads: int,
        alibi_slopes: Optional[List[float]],
        sliding_window: Optional[int],
        kv_cache_dtype: str,
        blocksparse_params: Optional[Dict[str, Any]] = None,
        logits_soft_cap: Optional[float] = None,
        attn_type: str = AttentionType.DECODER,
        kv_sharing_target_layer_name: Optional[str] = None,
        use_irope: bool = False,
    ) -> None:
        self.num_heads = num_heads
        self.head_size = head_size
        self.scale = torch.tensor(scale)
        self.num_kv_heads = num_kv_heads
        if alibi_slopes is not None:
            alibi_slopes = torch.tensor(alibi_slopes, dtype=torch.float32)
        self.alibi_slopes = alibi_slopes
        self.sliding_window = sliding_window
        self.kv_cache_dtype = kv_cache_dtype

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

    def split_kv_cache(
        self,
        kv_cache: torch.Tensor,
        num_kv_heads: int,
        head_size: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        key_cache = kv_cache[0]
        value_cache = kv_cache[1]
        return key_cache, value_cache

    def forward(
        self,
        layer: AttentionLayer,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: RBLNAttentionMetadata,
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
        query = query.view(b_size, self.num_kv_heads, self.num_queries_per_kv,
                           q_len, self.head_size)
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
            s = attn_metadata.seq_lens_tensor.to(torch.int16)[0][0]
            e = s + q_len
            block = attn_metadata.block_tables.to(torch.int16)[0]
            k_state = kv_cache[0][block].unsqueeze(0).slice_scatter(key,
                                                                    dim=3,
                                                                    start=s,
                                                                    end=e)
            v_state = kv_cache[1][block].unsqueeze(0).slice_scatter(value,
                                                                    dim=3,
                                                                    start=s,
                                                                    end=e)
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
                    attn_metadata.seq_lens_tensor.to(torch.int16),
                    attn_metadata.block_tables.to(torch.int16),
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
                    attn_metadata.seq_lens_tensor.to(torch.int16),
                    attn_metadata.block_tables.to(torch.int16),
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
