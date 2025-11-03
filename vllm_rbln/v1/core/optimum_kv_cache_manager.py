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

import torch
import math
from vllm.v1.core.kv_cache_manager import KVCacheBlocks, KVCacheManager

from dataclasses import dataclass
from typing import Literal, Optional, overload

from vllm.distributed.kv_events import KVCacheEvent
from vllm_rbln.logger import init_logger
from vllm.v1.core.kv_cache_coordinator import get_kv_cache_coordinator
from vllm.v1.core.kv_cache_utils import KVCacheBlock
from vllm.v1.kv_cache_interface import KVCacheConfig
from vllm.v1.metrics.stats import PrefixCacheStats
from vllm.v1.request import Request, RequestStatus
from vllm_rbln.v1.core.prefix_cache_manager.optimum_prefix_cache_manager import (
    RBLNPrefixKVCacheManager,
)

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
    ) -> None:
        """
        Initialize the RBLNKVCacheManager.
        It manages prefix_cache_manager in addition to the base KVCacheManager.
        It manages the mapping between inner blocks and outer blocks for prefix caching.
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
            assert attn_block_size is not None, \
                "`attn_block_size` must be provided when prefix caching is enabled."
            self.attn_block_size = attn_block_size
            self.prefix_cache_manager = RBLNPrefixKVCacheManager(
                ob_size=self.attn_block_size,
                ib_size=self.block_size,
                max_model_len=self.max_model_len,
                num_inner_blocks=self.block_pool.num_gpu_blocks - 1,
            )
        else:
            self.attn_block_size = self.block_size

    def free(self, request: Request, preemption: int =False) -> None:
        """Free the blocks allocated for the request.
        """
        if self.enable_caching:
            self.prefix_cache_manager.free_request(request.request_id, preemption=preemption)
        self.coordinator.free(request.request_id)

    def allocate_slots(
        self,
        request: Request,
        num_new_tokens: int,
        num_new_computed_tokens: int = 0,
        new_computed_blocks: Optional[KVCacheBlocks] = None,
    ) -> Optional[KVCacheBlocks]:
        """Add slots for a request with new tokens to append.

        Args:
            request: The request to allocate slots.
            num_new_tokens: The number of tokens to allocate, including external
                tokens. Note that this does not include tokens that have
                already been computed locally (i.e. new_computed_blocks).
            num_new_computed_tokens: The number of new computed tokens just
                hitting the prefix caching, excluding external tokens.
            new_computed_blocks: The cached blocks for the above new computed 
                tokens.
            num_lookahead_tokens: The number of speculative tokens to allocate.
                This is used by spec decode proposers with kv-cache such 
                as eagle.
            delay_cache_blocks: Whether to skip caching the blocks. This is
                used by P/D when allocating blocks used in a KV transfer
                which will complete in a future step.

        Blocks layout:
        ```
        -----------------------------------------------------------------------
        | < computed > | < new computed > |    < new >    | < pre-allocated > |
        -----------------------------------------------------------------------
        |                  < required >                   |
        --------------------------------------------------
        |                    < full >                  |
        ------------------------------------------------
                                          | <new full> |
                                          --------------
        ```
        The following *_blocks are illustrated in this layout.

        Returns:
            A list of new allocated blocks.
        """
        if num_new_tokens == 0:
            raise ValueError("num_new_tokens must be greater than 0")

        if new_computed_blocks is not None:
            new_computed_block_list = new_computed_blocks.blocks
        else:
            new_computed_block_list = tuple(
                [] for _ in range(len(self.kv_cache_config.kv_cache_groups)))
        # NOTE `new_computed_block_list` is used only for touch
        # When allocating new blocks, we do not re-use the provided
        # `new_computed_blocks`, we need to allocate new blocks.
        empty_computed_block_list = tuple(
            [] for _ in range(len(self.kv_cache_config.kv_cache_groups))
        )
        # Free the blocks that are skipped during the attention computation
        # (e.g., tokens outside the sliding window).
        # We can do this even if we cannot schedule this request due to
        # insufficient free blocks.
        # Should call this function before allocating new blocks to reduce
        # the number of evicted blocks.
        # self.coordinator.remove_skipped_blocks(request.request_id,
        #                                        request.num_computed_tokens)

        # The number of computed tokens is the number of computed tokens plus
        # the new prefix caching hits
        # In prefill, `num_computed_tokens` will be 0, and `num_new_tokens`
        # will the length of the input prompt.
        # In decode, `num_computed_tokens` will be the number of tokens of prompt
        # and generated tokens so far, 
        # and `num_new_tokens` will be 1.
        num_computed_tokens = request.num_computed_tokens
        num_tokens_need_slot = min(request.num_tokens, self.max_model_len)
        num_blocks_to_allocate = self.coordinator.get_num_blocks_to_allocate(
            request_id=request.request_id,
            num_tokens=num_tokens_need_slot,
            new_computed_blocks=empty_computed_block_list,
            num_encoder_tokens=0,
        )

        # print(f"[ib] request_id: {request.request_id} num_blocks_to_allocate: {num_blocks_to_allocate}")

        if num_blocks_to_allocate > self.block_pool.get_num_free_blocks():
            # Cannot allocate new blocks
            return None
        if self.enable_caching:
            # num_ibs_per_ob = self.attn_block_size // self.block_size
            # num_outer_blocks_to_allocate = math.ceil(num_blocks_to_allocate / num_ibs_per_ob)
            # if request.request_id == "2":
            # print(f"[ob] request_id: {request.request_id} num_outer_blocks_to_allocate: {num_outer_blocks_to_allocate}")
            if not self.prefix_cache_manager.can_allocate(
                num_blocks_to_allocate,
                num_computed_tokens,
            ):
                # Cannot allocate new outer blocks for prefix caching
                return None

        # Touch the computed blocks to make sure they won't be evicted.
        if self.enable_caching:
            self.block_pool.touch(new_computed_block_list)
        else:
            assert not any(new_computed_block_list), (
                "Computed blocks should be empty when "
                "prefix caching is disabled")

        # Append the new computed blocks to the request blocks until now to
        # avoid the case where the new blocks cannot be allocated.
        # TODO what is the role of the code?
        self.coordinator.save_new_computed_blocks(request.request_id,
                                                  empty_computed_block_list,
                                                )

        new_blocks = self.coordinator.allocate_new_blocks(
            request.request_id, num_tokens_need_slot, 0)

        # P/D: delay caching blocks if we have to recv from
        # remote. Update state for locally cached blocks.
        if not self.enable_caching:
            return KVCacheBlocks(new_blocks)

        # Allocate outer blocks for prefix caching.
        # TODO
        # if self.enable_caching:
        #     self.prefix_cache_manager.compute_num_blocks_to_allocate(new_blocks, num_computed_tokens)

        # NOTE(woosuk): We want to commit (cache) up to num_computed_tokens +
        # num_new_tokens, but must exclude "non-committable" tokens (e.g.,
        # draft tokens that could be rejected). Therefore, we cap the number
        # at `request.num_tokens`, ensuring only "finalized" tokens are cached.
        num_tokens_to_cache = min(num_computed_tokens + num_new_tokens,
                                  request.num_tokens)
        self.coordinator.cache_blocks(request, num_tokens_to_cache)
        return KVCacheBlocks(new_blocks)


    def get_prefix_cached_blocks_prefill(self, request: Request, cached_blocks: KVCacheBlocks,
        num_cached_tokens: int, inner_blocks: KVCacheBlocks) -> tuple[torch.Tensor, dict, int]:
        """Get the block table for prefill phase.
        """
        return self.prefix_cache_manager.get_block_table_prefill(
            request.request_id, cached_blocks,
            num_cached_tokens, inner_blocks
        )

    def get_prefix_cached_blocks_decode(self, request: Request, new_inner_blocks: KVCacheBlocks) -> torch.Tensor:
        """Get the block table for decode phase.
        """
        num_computed_tokens = request.num_computed_tokens
        return self.prefix_cache_manager.get_block_table_decode(request.request_id, num_computed_tokens, new_inner_blocks)