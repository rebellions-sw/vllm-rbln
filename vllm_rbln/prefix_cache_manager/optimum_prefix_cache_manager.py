# Copyright 2025 Rebellions Inc. All rights reserved.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at:

#     http://www.apache.org/licenses/LICENSE-2.0

import math
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from abc import ABC, abstractmethod
from collections import deque
from dataclasses import dataclass
from typing import Optional

import torch
from vllm.v1.core.kv_cache_manager import KVCacheBlocks

from vllm_rbln.logger import init_logger
from vllm_rbln.prefix_cache_manager.optimum_block_mapping_manager import (
    BlockMappingManager, RBLNBlock)
from vllm_rbln.prefix_cache_manager.optimum_eviction_policy import (
    FIFOEvictionPolicy, LRUEvictionPolicy)

logger = init_logger(__name__)


@dataclass
class BlockConfiguration:
    ob_size: int
    ib_size: int
    max_model_len: int
    num_ob: int

    def __post_init__(self):
        assert self.ob_size % self.ib_size == 0, \
            "ob_size must be a multiple of ib_size"

    @property
    def block_ratio(self) -> int:
        return self.ob_size // self.ib_size


@dataclass
class CacheSearchResult:
    cached_outer_blocks: list[int]
    cached_lengths: list[int]

    @property
    def has_cache_hit(self) -> bool:
        return len(self.cached_outer_blocks) > 0


class BlockAllocatorInterface(ABC):

    @abstractmethod
    def allocate(self, count: int) -> list[RBLNBlock]:
        pass

    @abstractmethod
    def deallocate(self, block: RBLNBlock) -> None:
        pass

    @abstractmethod
    def get_free_count(self) -> int:
        pass


class RBLNBlockAllocator(BlockAllocatorInterface):

    def __init__(self, num_blocks: int):
        self._free_blocks = deque([RBLNBlock(i) for i in range(num_blocks)])
        # Keep track of allocated blocks for validation
        self._allocated_blocks: dict[int, RBLNBlock] = {}

    def allocate(self, count: int) -> list[RBLNBlock]:
        if len(self._free_blocks) < count:
            raise RuntimeError(
                f"Insufficient free blocks. Requested: {count}, "
                f"Available: {len(self._free_blocks)}")

        allocated = []
        for _ in range(count):
            block = self._free_blocks.popleft()
            self._allocated_blocks[block.block_id] = block
            allocated.append(block)
        return allocated

    def deallocate(self, block: RBLNBlock) -> None:
        if block.block_id in self._allocated_blocks:
            self._free_blocks.append(block)
            del self._allocated_blocks[block.block_id]
        else:
            logger.warning("Attempting to deallocate unallocated block: %d",
                           block.block_id)

    def get_free_count(self) -> int:
        return len(self._free_blocks)

    def get_allocated_block(self, block_id: int) -> Optional[RBLNBlock]:
        return self._allocated_blocks.get(block_id)


class CacheSearchManager:
    """
    Search for cached blocks that can be reused.
    """

    def __init__(self, config: BlockConfiguration):
        self._config = config

    def find_cached_blocks(
            self, request_id: str, inner_blocks: list[int],
            mapping_manager: BlockMappingManager) -> CacheSearchResult:
        """
        Find cached outer blocks that match the given inner blocks.
        """
        best_match = self._try_match_request(inner_blocks, mapping_manager)

        if best_match.has_cache_hit:
            logger.debug("[PFX] [CACHE-HIT] REQUEST=%s hits OB=%s (IB=%s)",
                         request_id, best_match.cached_outer_blocks,
                         inner_blocks)

        return best_match

    def _calculate_cached_inner_blocks(self, cached_len_tokens: int) -> int:
        """
        Calculate the number of cached inner blocks
        based on the cached length in tokens.
        """
        return cached_len_tokens // self._config.ib_size

    def _try_match_request(
            self, cached_ib: list[int],
            mapping_manager: BlockMappingManager) -> CacheSearchResult:
        """
        Try to find the best matching outer blocks for the given inner blocks.
        NOTE Currently, we only support exact match of the inner blocks
        """
        cached_ob = []
        cached_lengths = []
        for start_ib_idx in range(0, len(cached_ib), self._config.block_ratio):
            end_ib_idx = min(start_ib_idx + self._config.block_ratio,
                             len(cached_ib))
            cur_ib_segment = cached_ib[start_ib_idx:end_ib_idx]
            candidate_obs = mapping_manager.get_outer_blocks_for_inner(
                cur_ib_segment[0])
            if len(candidate_obs) == 0:
                break
            for ob_id in candidate_obs:
                inner_blocks = mapping_manager.get_inner_blocks_for_outer(
                    ob_id)
                if inner_blocks[:len(cur_ib_segment)] == cur_ib_segment:
                    cached_ob.append(ob_id)
                    cached_lengths.append(
                        len(cur_ib_segment) * self._config.ib_size)
                    break

        return CacheSearchResult(cached_outer_blocks=cached_ob,
                                 cached_lengths=cached_lengths)


class MemoryPoolManager:
    """
    Manage a memory pool to return a tensor of block IDs.
    """

    def __init__(self, max_model_len: int, ob_size: int):
        self._pooled_tensor = torch.zeros(max_model_len // ob_size,
                                          dtype=torch.int32)

    def get_tensor_for_blocks(self, block_ids: list[int]) -> torch.Tensor:
        self._pooled_tensor.fill_(-1)

        if block_ids:
            value = torch.tensor(block_ids, dtype=torch.int32)
            self._pooled_tensor[:len(value)].copy_(value)

        return self._pooled_tensor.clone()


class RBLNPrefixKVCacheManager:

    def __init__(self, ob_size: int, ib_size: int, max_model_len: int,
                 num_ib: int):
        num_ob = num_ib // (ob_size // ib_size)
        self._config = BlockConfiguration(ob_size, ib_size, max_model_len,
                                          num_ob)
        self._allocator = RBLNBlockAllocator(num_ob)
        self._mapping_manager = BlockMappingManager()
        self._cache_search_manager = CacheSearchManager(self._config)
        self._memory_pool_manager = MemoryPoolManager(max_model_len, ob_size)
        self._eviction_policy = FIFOEvictionPolicy()

    def allocate_blocks(self, request_id: str, num_new_ob: int,
                        inner_blocks: list[int]) -> None:
        """
        Allocate blocks for a given request
        based on its phase (PREFILL or DECODE).
        """
        if self._mapping_manager.is_request_registered(request_id):
            self._handle_decode_allocation(request_id, num_new_ob,
                                           inner_blocks)
        else:
            self._handle_prefill_allocation(request_id, num_new_ob,
                                            inner_blocks)

    def _handle_decode_allocation(self, request_id: str, num_new_ob: int,
                                  inner_blocks: list[int]) -> None:
        """
        Allocate new blocks for DECODE phase.
        """
        if num_new_ob > 0:
            self._allocate_new_blocks(request_id, num_new_ob, inner_blocks)
        else:
            # Append to the last outer block
            request_blocks = self._mapping_manager.get_request_blocks(
                request_id)
            last_ob_id = request_blocks[-1]
            self._append_to_existing_block(last_ob_id, inner_blocks)

    def _handle_prefill_allocation(self, request_id: str, num_new_ob: int,
                                   inner_blocks: list[int]) -> None:
        """
        Allocate new blocks for PREFILL phase.
        """
        self._allocate_new_blocks(request_id, num_new_ob, inner_blocks)

    def _allocate_new_blocks(self, request_id: str, num_new_ob: int,
                             inner_blocks: list[int]) -> None:
        """
        Allocate new outer blocks and create mappings.
        """
        assert num_new_ob > 0, "One or more new blocks must be allocated"
        # Allocate blocks
        new_blocks = self._allocator.allocate(num_new_ob)

        # Create mappings
        block_ids = []
        for i, block in enumerate(new_blocks):
            start_idx = i * self._config.block_ratio
            end_idx = min((i + 1) * self._config.block_ratio,
                          len(inner_blocks))
            block_inner_blocks = inner_blocks[start_idx:end_idx]

            self._mapping_manager.create_mapping(block, block_inner_blocks,
                                                 request_id)
            self._eviction_policy.register_block(block.block_id)
            block_ids.append(block.block_id)

        logger.debug("[PFX] [ALLOC] REQUEST=%s OB=%s (IB=%s)", request_id,
                     block_ids, inner_blocks)

    def compute_num_blocks_to_allocate(self,
                                       inner_blocks: list[int],
                                       num_allocated_tokens: int = 0) -> int:
        """
        Compute the number of outer blocks to allocate based on inner blocks
        and the number of already allocated tokens.
        """
        if len(inner_blocks) == 0:
            return 0

        if num_allocated_tokens == 0:
            # PREFILL
            num_obs_needed = math.ceil(
                len(inner_blocks) / self._config.block_ratio)
        else:
            # DECODE
            num_already_allocated_ibs = \
                num_allocated_tokens // self._config.ib_size

            if num_already_allocated_ibs % self._config.block_ratio == 0:
                num_obs_needed = 1
            else:
                num_obs_needed = 0

        return num_obs_needed

    def can_allocate(self, new_blocks: list[int],
                     num_computed_tokens: int) -> bool:
        # 1. Check if the enough outer blocks are free
        required_num_ob = self.compute_num_blocks_to_allocate(
            new_blocks, num_computed_tokens)
        free_count = self._allocator.get_free_count()
        if free_count >= required_num_ob:
            return True
        # 2. Check if we can evict enough blocks
        evict_count = required_num_ob - free_count
        can_evict = self._eviction_policy.can_evict(self._mapping_manager,
                                                    evict_count)
        return can_evict

    def ensure_free_blocks(self, num_new_blocks: int) -> None:
        """
        If there are not enough free blocks,
        evict some blocks based on the eviction policy.
        """
        free_count = self._allocator.get_free_count()
        if free_count >= num_new_blocks:
            return

        evict_count = num_new_blocks - free_count
        blocks_to_evict = self._eviction_policy.select_blocks_for_eviction(
            self._mapping_manager, evict_count)
        # Check if we could evict enough blocks
        if len(blocks_to_evict) < evict_count:
            raise RuntimeError(
                f"Cannot evict enough blocks. Need {evict_count}, "
                f"can evict {len(blocks_to_evict)}")
        # NOTE: In vLLM, the blocks are returned
        # to the free block queue in reversed order.
        # It is for preventing memory fragmentation.
        # But here, we don't need to reverse the order.
        for block_id in blocks_to_evict:
            self._evict_block(block_id)

    def _evict_block(self, block_id: int) -> None:
        """
        Evict a block and free its resources.
        """
        mapping = self._mapping_manager.remove_mapping(block_id)
        if mapping:
            block = self._allocator.get_allocated_block(block_id)
            if block:
                self._allocator.deallocate(block)
                self._eviction_policy.unregister_block(block_id)
                logger.debug("[PFX] [EVICTION] OB=%d (IB=%s)", block_id,
                             mapping.inner_block_ids)
            else:
                logger.error("Block %d not found in allocator during eviction",
                             block_id)

    def _append_to_existing_block(self, outer_block_id: int,
                                  inner_blocks: list[int]) -> None:
        """
        Add inner blocks to an existing outer block.
        """
        assert len(
            inner_blocks) == 1, "Can only append one inner block at a time"
        ib_id = inner_blocks[0]

        # Update the outer to inner mapping
        mapping = self._mapping_manager.get_mapping(outer_block_id)
        if not mapping:
            raise RuntimeError(
                f"Mapping not found for outer block {outer_block_id}")

        mapping.inner_block_ids.append(ib_id)

        # Update the inner to outer mapping
        self._mapping_manager.add_new_inner_to_outer(ib_id, outer_block_id)

    def free_request(self, request_id: str, preemption: bool = False) -> None:
        """
        Called when a request is completed or preempted.
        Free all blocks associated with a given request.
        If it is not preemption, keep the mappings for potential future reuse.
        """
        outer_blocks = self._mapping_manager.remove_request(request_id)

        for block_id in outer_blocks:
            mapping = self._mapping_manager.get_mapping(block_id)
            if mapping:
                mapping.is_active = False
                mapping.request_id = None
                if preemption:
                    self._evict_block(block_id)

    def get_matched_outer_blocks(
            self, request_id: str,
            inner_blocks: list[int]) -> tuple[list[int], list[int]]:
        """
        Get the matched outer blocks using inner blocks.
        """
        result = self._cache_search_manager.find_cached_blocks(
            request_id, inner_blocks, self._mapping_manager)

        if result.has_cache_hit and isinstance(self._eviction_policy,
                                               LRUEvictionPolicy):
            for ob_id in result.cached_outer_blocks:
                self._eviction_policy.touch(ob_id)

        return result.cached_outer_blocks, result.cached_lengths

    def get_blocks(self, request_id: str) -> torch.Tensor:
        """
        Get the tensor of outer block IDs for a given request.
        """
        if not self._mapping_manager.is_request_registered(request_id):
            logger.warning("Request %s not found in mappings", request_id)
            return self._memory_pool_manager.get_tensor_for_blocks([])

        block_ids = self._mapping_manager.get_request_blocks(request_id)
        return self._memory_pool_manager.get_tensor_for_blocks(block_ids)

    def get_block_table_prefill(
        self, request_id: str, cached_blocks: KVCacheBlocks,
        num_cached_tokens: int, inner_blocks: KVCacheBlocks
    ) -> tuple[torch.Tensor, list[int], list[int]]:

        inner_blocks = inner_blocks.get_block_ids()[0]
        cached_blocks = cached_blocks.get_block_ids()[0]

        if len(inner_blocks) == 0:
            return self.get_blocks(request_id), [], []

        num_new_ob = self.compute_num_blocks_to_allocate(inner_blocks, 0)
        self.ensure_free_blocks(num_new_ob)
        cached_block_table, cached_length = self.get_matched_outer_blocks(
            request_id, cached_blocks)
        if sum(cached_length) < num_cached_tokens:
            logger.debug(
                "The blocks %s is not in RBLN prefix cache manager. "
                "Falling back to full attention", cached_blocks)
        self.allocate_blocks(request_id, num_new_ob, inner_blocks)
        return self.get_blocks(request_id), cached_block_table, cached_length

    def get_block_table_decode(self, request_id: str,
                               num_allocated_tokens: int,
                               inner_blocks: KVCacheBlocks) -> torch.Tensor:
        inner_blocks = inner_blocks.get_block_ids()[0]

        if len(inner_blocks) == 0:
            return self.get_blocks(request_id)

        num_new_ob = self.compute_num_blocks_to_allocate(
            inner_blocks, num_allocated_tokens)
        self.ensure_free_blocks(num_new_ob)
        self.allocate_blocks(request_id, num_new_ob, inner_blocks)
        return self.get_blocks(request_id)
