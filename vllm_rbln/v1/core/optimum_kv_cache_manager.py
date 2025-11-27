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

from typing import Optional

import torch
from vllm.v1.core.kv_cache_manager import KVCacheBlocks, KVCacheManager
from vllm.v1.kv_cache_interface import KVCacheConfig
from vllm.v1.request import Request

from vllm_rbln.logger import init_logger
from vllm_rbln.v1.core.prefix_cache_manager import RBLNPrefixKVCacheManager

logger = init_logger(__name__)


class RBLNKVCacheManager(KVCacheManager):

    def __init__(
        self,
        kv_cache_config: KVCacheConfig,
        max_model_len: int,
        enable_caching: bool = True,
        use_eagle: bool = False,
        log_stats: bool = False,
        enable_kv_cache_events: bool = False,
        dcp_world_size: int = 1,
        attn_block_size: Optional[int] = None,
        max_num_seqs: int = 1,
    ) -> None:
        """
        RBLNKVCacheManager = KVCacheManager + PrefixKVCacheManager.
        PrefixKVCacheManager manages the mapping
        between inner blocks and outer blocks for prefix caching.
        """
        super().__init__(
            kv_cache_config,
            max_model_len,
            enable_caching,
            use_eagle,
            log_stats,
            enable_kv_cache_events,
            dcp_world_size,
        )
        if enable_caching:
            self.prefix_cache_manager = RBLNPrefixKVCacheManager(
                ob_size=attn_block_size,
                ib_size=self.block_size,
                max_model_len=self.max_model_len,
                max_num_seqs=max_num_seqs,
                num_inner_blocks=self.block_pool.num_gpu_blocks - 1,
            )

    def free(self, request: Request, preemption: int = False) -> None:
        """Free the blocks allocated for the request.
        """
        if self.enable_caching:
            self.prefix_cache_manager.free_request(request.request_id,
                                                   preemption=preemption)
        self.coordinator.free(request.request_id)

    def allocate_slots(
        self,
        request: Request,
        num_new_tokens: int,
        num_new_computed_tokens: int = 0,
        new_computed_blocks: Optional[KVCacheBlocks] = None,
        num_lookahead_tokens: int = 0,
        delay_cache_blocks: bool = False,
        num_encoder_tokens: int = 0,
    ) -> Optional[KVCacheBlocks]:
        assert num_lookahead_tokens == 0
        assert not delay_cache_blocks
        assert num_encoder_tokens == 0

        if num_new_tokens == 0:
            raise ValueError("num_new_tokens must be greater than 0")

        if new_computed_blocks is not None:
            new_computed_block_list = new_computed_blocks.blocks
        else:
            new_computed_block_list = tuple(
                [] for _ in range(len(self.kv_cache_config.kv_cache_groups)))
        # NOTE `new_computed_block_list` is used only for touch
        # When allocating new blocks, we do not reuse the provided
        # `new_computed_blocks` and we need to allocate new blocks.
        empty_computed_block_list = tuple(
            [] for _ in range(len(self.kv_cache_config.kv_cache_groups)))

        # In prefill,
        # `num_computed_tokens` = 0,
        # `num_new_tokens` = the length of the input prompt.
        # In decode,
        # `num_computed_tokens` = the length of prompt + generated text
        # `num_new_tokens` = 1.
        is_prefill = (request.num_computed_tokens == 0)
        num_computed_tokens = request.num_computed_tokens
        num_tokens_need_slot = min(request.num_tokens, self.max_model_len)
        num_blocks_to_allocate = self.coordinator.get_num_blocks_to_allocate(
            request_id=request.request_id,
            num_tokens=num_tokens_need_slot,
            new_computed_blocks=empty_computed_block_list,
            num_encoder_tokens=0,
        )

        if num_blocks_to_allocate > self.block_pool.get_num_free_blocks():
            # Cannot allocate new blocks
            return None

        # Edge case
        # removed_blocks refers the blocks
        # that are removed from the free blocks
        # after the touch function is called
        # We need to check the free blocks will be enough
        # after the touch function is called
        # if self.enable_caching:
        #     removed_blocks = 0
        #     for blocks_per_group in new_computed_block_list:
        #         for block in blocks_per_group:
        #             if block.ref_cnt == 0 and not block.is_null:
        #                 removed_blocks += 1

        #     if num_blocks_to_allocate + removed_blocks > \
        #         self.block_pool.get_num_free_blocks():
        #         return None

        if self.enable_caching and \
            not self.prefix_cache_manager.can_allocate(
                    num_blocks_to_allocate,
                    num_computed_tokens,
            ):
            # Cannot allocate new outer blocks for prefix caching
            return None

        # TODO (eunji): The touch function
        # increases the ref_cnt. We don't need ref_cnt
        # because we don't reuse the provided computed blocks
        # and just copy the prefix matched blocks.
        # But for consistency with original vllm and
        # cache hit rate, we keep the touch function here.
        # It triggers all blocks are not freed
        # even though the request is freed.
        # if self.enable_caching:
        #     self.block_pool.touch(new_computed_block_list)
        # else:
        #     assert not any(new_computed_block_list), (
        #         "Computed blocks should be empty when "
        #         "prefix caching is disabled")

        # Generate req_to_blocks, num_cached_block
        # in the coordinator
        # `empty_computed_block_list` is used here to avoid
        # saving the computed blocks to the request state
        self.coordinator.save_new_computed_blocks(
            request.request_id,
            empty_computed_block_list,
        )

        new_blocks = self.coordinator.allocate_new_blocks(
            request.request_id, num_tokens_need_slot, 0)

        # P/D: delay caching blocks if we have to recv from
        # remote. Update state for locally cached blocks.
        if not self.enable_caching:
            return KVCacheBlocks(new_blocks)

        # Allocate outer blocks for prefix caching
        # following the inner blocks allocation
        inner_block_ids = [block.block_id for block in new_blocks[0]]
        cached_blocks = [
            block.block_id for block in new_computed_block_list[0]
        ]
        self.prefix_cache_manager.allocate_blocks(
            request.request_id,
            num_computed_tokens,
            inner_block_ids,
        )
        # Set the computed blocks as cached blocks
        self._set_prefix_cached_blocks(
            request.request_id,
            num_new_computed_tokens,
            cached_blocks,
            is_prefill,
        )

        # Set the newly allocated blocks as cached blocks
        self._set_prefix_cached_blocks(
            request.request_id,
            num_new_computed_tokens,
            inner_block_ids,
            is_prefill,
        )

        # NOTE(woosuk): We want to commit (cache) up to num_computed_tokens +
        # num_new_tokens, but must exclude "non-committable" tokens (e.g.,
        # draft tokens that could be rejected). Therefore, we cap the number
        # at `request.num_tokens`, ensuring only "finalized" tokens are cached.
        num_tokens_to_cache = min(num_computed_tokens + num_new_tokens,
                                  request.num_tokens)
        self.coordinator.cache_blocks(request, num_tokens_to_cache)
        return KVCacheBlocks(new_blocks)

    def get_prefix_cached_blocks(
        self,
        request: Request,
        new_computed_blocks: KVCacheBlocks,
        num_new_computed_tokens: int,
    ) -> tuple[list[int], list[int]]:
        cached_blocks = new_computed_blocks.get_block_ids()[0]
        cached_block_table, cached_length = \
            self.prefix_cache_manager.get_matched_outer_blocks(
            request.request_id, cached_blocks,
            num_new_computed_tokens,
        )

        return cached_block_table, cached_length

    def get_block_table(self, request_id: str) -> torch.Tensor:
        return self.prefix_cache_manager.get_blocks(request_id)

    def get_dummy_block(self) -> int:
        return self.prefix_cache_manager.get_dummy_block()

    def _set_prefix_cached_blocks(
        self,
        request_id: str,
        num_new_computed_tokens: int,
        cached_blocks: list[int],
        is_prefill: bool,
    ) -> None:
        # NOTE Currently, this function is called only prefill
        # and prefix caching is hit in the original
        # kv cache manager.
        if not is_prefill:
            return
        allocated_outer_blocks = self.prefix_cache_manager.get_block_ids(
            request_id)
        self.prefix_cache_manager.set_cached_blocks(
            cached_blocks,
            allocated_outer_blocks,
        )
