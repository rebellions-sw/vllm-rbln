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
from abc import ABC, abstractmethod
from collections import deque
from dataclasses import dataclass
from typing import Optional

import torch

from vllm_rbln.logger import init_logger

from .optimum_block_mapping_manager import BlockMappingManager, RBLNBlock
from .optimum_eviction_policy import LRUEvictionPolicy

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
    It maintains a per-request cache of search results
    until the request is freed.
    """

    def __init__(self, config: BlockConfiguration):
        self._config = config
        self._cached_blocks_per_request: dict[str, CacheSearchResult] = {}

    def find_cached_blocks(
            self, request_id: str, cached_blocks: list[int],
            num_new_computed_tokens: int,
            mapping_manager: BlockMappingManager) -> CacheSearchResult:
        """
        Find cached outer blocks that match the given inner blocks.
        """
        if request_id in self._cached_blocks_per_request:
            cached_outer_blocks = self._cached_blocks_per_request[
                request_id].cached_outer_blocks
            if self._all_valid(cached_outer_blocks, mapping_manager):
                return self._cached_blocks_per_request[request_id]

        best_match = self._try_match_request(cached_blocks, mapping_manager)
        self._cached_blocks_per_request[request_id] = best_match
        final_num_cached_tokens = sum(best_match.cached_lengths)
        if final_num_cached_tokens < num_new_computed_tokens:
            logger.debug(
                "The request %s cannot hit "
                "all new computed tokens(%d). "
                "The last tokens %d are already evicted.", request_id,
                num_new_computed_tokens,
                num_new_computed_tokens - final_num_cached_tokens)

        if best_match.has_cache_hit:
            logger.debug("[PFX] [CACHE-HIT] REQUEST=%s hits OB=%s (IB=%s)",
                         request_id, best_match.cached_outer_blocks,
                         cached_blocks)

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
            # TODO return outer blocks
            #   1. exact match
            #   2. not exact match, but includes the matched inner blocks
            ob_id = mapping_manager.get_outer_block_for_inner(
                cur_ib_segment[0])
            print("cur_ib_segment:", cur_ib_segment)
            print("ob_id:", ob_id)
            if ob_id is not None:
                inner_blocks = mapping_manager.get_inner_blocks_for_outer(
                    ob_id)
                num_cached_ibs = len(cur_ib_segment)
                if inner_blocks[:num_cached_ibs] == cur_ib_segment:
                    cached_ob.append(ob_id)
                    cached_lengths.append(num_cached_ibs *
                                          self._config.ib_size)
            else:
                ob_id, num_cached_ibs = \
                    mapping_manager.get_outer_block_for_copied_inner(
                    cur_ib_segment)
                if num_cached_ibs == 0:
                    break
                cached_ob.append(ob_id)
                cached_lengths.append(num_cached_ibs * self._config.ib_size)
        print("cached_ob:", cached_ob)
        print("cached_lengths:", cached_lengths)
        return CacheSearchResult(cached_outer_blocks=cached_ob,
                                 cached_lengths=cached_lengths)

    def remove_cached_blocks(self, request_id: str) -> None:
        self._cached_blocks_per_request.pop(request_id, None)

    def _all_valid(self, cached_outer_blocks: list[int],
                   mapping_manager: BlockMappingManager) -> bool:
        for ob_id in cached_outer_blocks:
            if mapping_manager.get_mapping(ob_id) is None:
                return False
        return True


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
                 num_inner_blocks: int):
        assert ob_size % ib_size == 0, \
            "Outer block size must be a multiple of inner block size"
        num_ob = math.ceil(num_inner_blocks / (ob_size // ib_size))
        self._config = BlockConfiguration(ob_size, ib_size, max_model_len,
                                          num_ob)
        self._allocator = RBLNBlockAllocator(num_ob)
        self._mapping_manager = BlockMappingManager()
        self._cache_search_manager = CacheSearchManager(self._config)
        self._memory_pool_manager = MemoryPoolManager(max_model_len, ob_size)
        self._eviction_policy = LRUEvictionPolicy()

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

    def _compute_num_blocks_to_allocate(self,
                                        num_inner_blocks: int,
                                        num_allocated_tokens: int = 0) -> int:
        """
        Compute the number of outer blocks to allocate based on inner blocks
        and the number of already allocated tokens.
        """
        if num_inner_blocks == 0:
            return 0

        if num_allocated_tokens == 0:
            # PREFILL
            num_obs_needed = math.ceil(num_inner_blocks /
                                       self._config.block_ratio)
        else:
            # DECODE
            num_already_allocated_ibs = \
                num_allocated_tokens // self._config.ib_size

            if num_already_allocated_ibs % self._config.block_ratio == 0:
                num_obs_needed = 1
            else:
                num_obs_needed = 0

        return num_obs_needed

    def _check_free_blocks(self, num_new_blocks: int) -> None:
        """
        Check that there are enough free blocks to allocate.
        """
        free_count = self._allocator.get_free_count()
        assert free_count >= num_new_blocks, \
            f"Free blocks ({free_count}) should be " \
            f"greater than or equal to {num_new_blocks}"

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
        self._cache_search_manager.remove_cached_blocks(request_id)

        for block_id in outer_blocks:
            mapping = self._mapping_manager.get_mapping(block_id)
            if mapping:
                mapping.is_active = False
                mapping.request_id = None
                if preemption:
                    self._evict_block(block_id)

    def get_matched_outer_blocks(
            self, request_id: str, cached_blocks: list[int],
            num_new_computed_tokens: int) -> tuple[list[int], list[int]]:
        """
        Get the matched outer blocks using inner blocks.
        """
        result = self._cache_search_manager.find_cached_blocks(
            request_id, cached_blocks, num_new_computed_tokens,
            self._mapping_manager)

        if result.has_cache_hit and isinstance(self._eviction_policy,
                                               LRUEvictionPolicy):
            for ob_id in result.cached_outer_blocks:
                self._eviction_policy.touch(ob_id)

        return result.cached_outer_blocks, result.cached_lengths

    def get_blocks(self, request_id: str) -> torch.Tensor:
        """
        Get the tensor of outer block IDs for a given request.
        """
        block_ids = self.get_block_ids(request_id)
        return self._memory_pool_manager.get_tensor_for_blocks(block_ids)

    def get_block_ids(self, request_id: str) -> list[int]:
        """
        Get the list of outer block IDs for a given request.
        """
        if not self._mapping_manager.is_request_registered(request_id):
            logger.warning("Request %s not found in mappings", request_id)
            return []

        return self._mapping_manager.get_request_blocks(request_id)

    def can_allocate(self, num_new_blocks: int,
                     num_computed_tokens: int) -> bool:
        # 1. Check if the enough outer blocks are free
        required_num_ob = self._compute_num_blocks_to_allocate(
            num_new_blocks, num_computed_tokens)
        free_count = self._allocator.get_free_count()
        if free_count >= required_num_ob:
            return True

        # 2. Check if we can evict enough blocks
        evict_count = required_num_ob - free_count
        blocks_to_evict = self._eviction_policy.select_blocks_for_eviction(
            self._mapping_manager, evict_count)

        can_evict = len(blocks_to_evict) >= evict_count

        # 3. Evict blocks if possible
        if can_evict:
            # NOTE: In vLLM, the blocks are returned
            # to the free block queue in reversed order.
            # It is for preventing memory fragmentation.
            # But here, we don't need to reverse the order.
            for block_id in blocks_to_evict:
                self._evict_block(block_id)
        return can_evict

    def allocate_blocks(self, request_id: str, num_allocated_tokens: int,
                        inner_blocks: list[int]) -> None:
        """
        Allocate blocks for a given request
        based on its phase (PREFILL or DECODE).
        """
        num_new_ib = len(inner_blocks)
        if num_new_ib == 0:
            return

        num_new_ob = self._compute_num_blocks_to_allocate(
            num_new_ib, num_allocated_tokens)

        self._check_free_blocks(num_new_ob)

        if self._mapping_manager.is_request_registered(request_id):
            self._handle_decode_allocation(request_id, num_new_ob,
                                           inner_blocks)
        else:
            self._handle_prefill_allocation(request_id, num_new_ob,
                                            inner_blocks)

    def set_cached_blocks(self, cached_blocks: list[int],
                          allocated_outer_blocks: list[int]) -> None:
        """
        Mark the blocks associated with the request as cached.
        """
        self._mapping_manager.match_cached_blocks(cached_blocks,
                                                  allocated_outer_blocks,
                                                  self._config.block_ratio)
