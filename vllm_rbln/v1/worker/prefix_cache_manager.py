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

from abc import ABC, abstractmethod
from collections import deque
from dataclasses import dataclass
from typing import Optional, dict, list, tuple

import torch

from vllm_rbln.logger import init_logger

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
class BlockMapping:
    outer_block_id: int
    inner_block_ids: list[int]
    request_id: Optional[str] = None
    is_active: bool = True


@dataclass
class CacheSearchResult:
    cached_outer_blocks: list[int]
    cached_lengths: list[int]
    source_request_id: Optional[str] = None

    @property
    def has_cache_hit(self) -> bool:
        return len(self.cached_outer_blocks) > 0


class RBLNBlock:

    def __init__(self, block_id: int):
        self.block_id = block_id

    def __repr__(self) -> str:
        return f"RBLNBlock(id={self.block_id})"


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

    def allocate(self, count: int) -> list[RBLNBlock]:
        if len(self._free_blocks) < count:
            raise RuntimeError(
                f"Insufficient free blocks. Requested: {count}, "
                "Available: {len(self._free_blocks)}")

        return [self._free_blocks.popleft() for _ in range(count)]

    def deallocate(self, block: RBLNBlock) -> None:
        self._free_blocks.append(block)

    def get_free_count(self) -> int:
        return len(self._free_blocks)


class BlockMappingManager:
    """
    Manage mappings between outer blocks and inner blocks.
    Also manage mappings between requests and their allocated outer blocks.
    """

    def __init__(self):
        self._outer_to_inner: dict[int, BlockMapping] = {}
        self._inner_to_outer: dict[int, list[int]] = {}
        self._request_mappings: dict[str, list[int]] = {}

    def create_mapping(self, outer_block: RBLNBlock, inner_blocks: list[int],
                       request_id: str) -> None:
        """
        Create a new block mapping.
        """
        mapping = BlockMapping(outer_block_id=outer_block.block_id,
                               inner_block_ids=inner_blocks.copy(),
                               request_id=request_id)

        self._outer_to_inner[outer_block.block_id] = mapping

        # Update Inner to outer mapping
        for ib_id in inner_blocks:
            if ib_id not in self._inner_to_outer:
                self._inner_to_outer[ib_id] = []
            self._inner_to_outer[ib_id].append(outer_block.block_id)

        # Update Request mapping
        if request_id not in self._request_mappings:
            self._request_mappings[request_id] = []
        self._request_mappings[request_id].append(outer_block.block_id)

    def remove_mapping(self, outer_block_id: int) -> Optional[BlockMapping]:
        """
        Remove a mapping by outer block ID and return the removed mapping.
        """
        mapping = self._outer_to_inner.pop(outer_block_id, None)
        if mapping:
            for ib_id in mapping.inner_block_ids:
                if ib_id in self._inner_to_outer:
                    self._inner_to_outer[ib_id].remove(outer_block_id)
                    if not self._inner_to_outer[ib_id]:
                        del self._inner_to_outer[ib_id]

        return mapping

    def get_request_blocks(self, request_id: str) -> list[int]:
        """
        Return the list of outer block IDs associated with a request.
        """
        return self._request_mappings.get(request_id, []).copy()

    def remove_request(self, request_id: str) -> list[int]:
        """
        Remove all mappings associatedwith a request
        and return the outer block IDs.
        """
        return self._request_mappings.pop(request_id, [])

    def get_mapping(self, outer_block_id: int) -> Optional[BlockMapping]:
        """
        Return the mapping for a given outer block ID.
        """
        return self._outer_to_inner.get(outer_block_id)

    def get_inactive_mappings(self) -> list[BlockMapping]:
        """
        Return a list of inactive mappings.
        """
        return [
            mapping for mapping in self._outer_to_inner.values()
            if not mapping.is_active
        ]


class CacheSearchManager:
    """
    Search for cached blocks that can be reused.
    """

    def __init__(self, config: BlockConfiguration):
        self._config = config

    def find_cached_blocks(
            self, request_id: str, num_computed_tokens: int,
            inner_blocks: list[int],
            mapping_manager: BlockMappingManager) -> CacheSearchResult:
        """
        Find cached outer blocks that match the given inner blocks.
        """
        num_cached_ib = self._calculate_cached_inner_blocks(
            num_computed_tokens)
        if num_cached_ib == 0:
            return CacheSearchResult([], [])

        cached_ib = inner_blocks[:num_cached_ib]
        best_match = CacheSearchResult([], [])

        # Find the cached blocks in unit of requests
        # NOTE we don't match cached blocks in unit of outer blocks
        for req_id in mapping_manager._request_mappings:
            if req_id == request_id:
                continue

            match_result = self._try_match_request(req_id, cached_ib,
                                                   mapping_manager)

            if len(match_result.cached_outer_blocks) > len(
                    best_match.cached_outer_blocks):
                best_match = match_result

        if best_match.has_cache_hit:
            logger.debug(
                "[PFX] [CACHE-HIT] REQUEST=%s -> REQUEST=%s (IB=%s of OB=%s)",
                request_id, best_match.source_request_id, cached_ib,
                best_match.cached_outer_blocks)

        return best_match

    def _calculate_cached_inner_blocks(self, cached_len_tokens: int) -> int:
        """
        Calculate the number of cached inner blocks
        based on the cached length in tokens.
        """
        return cached_len_tokens // self._config.ib_size

    def _try_match_request(
            self, target_request_id: str, cached_ib: list[int],
            mapping_manager: BlockMappingManager) -> CacheSearchResult:
        """
        Check if the cached inner blocks match
        with the blocks of the target request.
        """
        cached_ob = []
        cached_len = []

        request_blocks = mapping_manager.get_request_blocks(target_request_id)

        for ob_idx, ob_id in enumerate(request_blocks):
            start_pos = ob_idx * self._config.block_ratio
            end_pos = min((ob_idx + 1) * self._config.block_ratio,
                          len(cached_ib))

            if start_pos >= len(cached_ib):
                break

            cur_cached_ib = cached_ib[start_pos:end_pos]
            mapping = mapping_manager.get_mapping(ob_id)

            if not mapping or len(
                    mapping.inner_block_ids) < len(cur_cached_ib):
                break

            if mapping.inner_block_ids[:len(cur_cached_ib)] == cur_cached_ib:
                cached_ob.append(ob_id)
                cached_len.append(len(cur_cached_ib) * self._config.ib_size)
            else:
                break

        return CacheSearchResult(cached_outer_blocks=cached_ob,
                                 cached_lengths=cached_len,
                                 source_request_id=target_request_id)


class EvictionPolicy:
    """
    Simple eviction policy to select blocks for eviction.
    TODO upgrade to LRU or LFU based policy.
    """

    def select_blocks_for_eviction(self, mapping_manager: BlockMappingManager,
                                   count: int) -> list[int]:
        # Select blocks for eviction.
        inactive_mappings = mapping_manager.get_inactive_mappings()
        evicted_blocks = []

        for mapping in inactive_mappings:
            if len(evicted_blocks) >= count:
                break
            evicted_blocks.append(mapping.outer_block_id)

        return evicted_blocks


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
                 num_ob: int):
        self._config = BlockConfiguration(ob_size, ib_size, max_model_len,
                                          num_ob)
        self._allocator = RBLNBlockAllocator(num_ob)
        self._mapping_manager = BlockMappingManager()
        self._cache_search_manager = CacheSearchManager(self._config)
        self._memory_pool_manager = MemoryPoolManager(max_model_len, ob_size)
        self._eviction_policy = EvictionPolicy()

    def allocate_blocks(self, request_id: str, cached_len: int,
                        inner_blocks: list[int]) -> None:
        """
        Allocate blocks for a given request
        based on its phase (PREFILL or DECODE).
        """
        if request_id in self._mapping_manager._request_mappings:
            self._handle_decode_allocation(request_id, cached_len,
                                           inner_blocks)
        else:
            self._handle_prefill_allocation(request_id, inner_blocks)

    def _handle_decode_allocation(self, request_id: str, cached_len: int,
                                  inner_blocks: list[int]) -> None:
        """
        Allocate new blocks for DECODE phase.
        """
        num_already_allocated_ibs = cached_len // self._config.ib_size

        if num_already_allocated_ibs % self._config.block_ratio == 0:
            # 새로운 outer block 필요
            self._allocate_new_blocks(request_id, 1, inner_blocks)
        else:
            # 기존 마지막 outer block에 추가
            request_blocks = self._mapping_manager.get_request_blocks(
                request_id)
            last_ob_id = request_blocks[-1]
            self._append_to_existing_block(last_ob_id, inner_blocks)

    def _handle_prefill_allocation(self, request_id: str,
                                   inner_blocks: list[int]) -> None:
        """
        Allocate new blocks for PREFILL phase.
        """
        num_obs_needed = (len(inner_blocks) + self._config.block_ratio -
                          1) // self._config.block_ratio
        self._allocate_new_blocks(request_id, num_obs_needed, inner_blocks)

    def _allocate_new_blocks(self, request_id: str, count: int,
                             inner_blocks: list[int]) -> None:
        """
        Allocate new outer blocks and create mappings.
        """
        # Ensure enough free blocks
        self._ensure_free_blocks(count)

        # Allocate blocks
        new_blocks = self._allocator.allocate(count)

        # Create mappings
        for i, block in enumerate(new_blocks):
            start_idx = i * self._config.block_ratio
            end_idx = min((i + 1) * self._config.block_ratio,
                          len(inner_blocks))
            block_inner_blocks = inner_blocks[start_idx:end_idx]

            self._mapping_manager.create_mapping(block, block_inner_blocks,
                                                 request_id)

        block_ids = [block.block_id for block in new_blocks]
        logger.debug("[PFX] [ALLOC] REQUEST=%s OB=%s (IB=%s)", request_id,
                     block_ids, inner_blocks)

    def _ensure_free_blocks(self, needed_count: int) -> None:
        """
        If there are not enough free blocks,
        evict some blocks based on the eviction policy.
        """
        free_count = self._allocator.get_free_count()
        if free_count >= needed_count:
            return

        evict_count = needed_count - free_count
        blocks_to_evict = self._eviction_policy.select_blocks_for_eviction(
            self._mapping_manager, evict_count)

        for block_id in blocks_to_evict:
            self._evict_block(block_id)

    def _evict_block(self, block_id: int) -> None:
        """
        Evict a block and free its resources.
        """
        mapping = self._mapping_manager.remove_mapping(block_id)
        if mapping:
            block = RBLNBlock(block_id)  # mock block object
            self._allocator.deallocate(block)
            logger.debug("[PFX] [EVICTION] OB=%d (IB=%s)", block_id,
                         mapping.inner_block_ids)

    def _append_to_existing_block(self, outer_block_id: int,
                                  inner_blocks: list[int]) -> None:
        """
        Add inner blocks to an existing outer block.
        """
        assert len(
            inner_blocks) == 1, "Can only append one inner block at a time"

        mapping = self._mapping_manager.get_mapping(outer_block_id)
        if mapping:
            mapping.inner_block_ids.extend(inner_blocks)

            # Update the inner to outer mapping
            ib_id = inner_blocks[0]
            if ib_id not in self._mapping_manager._inner_to_outer:
                self._mapping_manager._inner_to_outer[ib_id] = []
            self._mapping_manager._inner_to_outer[ib_id].append(outer_block_id)

    def free_request(self, request_id: str) -> None:
        """
        Free all blocks associated with a given request.
        But keep the mappings for potential future reuse.
        """
        outer_blocks = self._mapping_manager.remove_request(request_id)

        for block_id in outer_blocks:
            mapping = self._mapping_manager.get_mapping(block_id)
            if mapping:
                mapping.is_active = False
                mapping.request_id = None

        logger.debug("[PFX] [FREE] REQUEST=%s OB=%s", request_id, outer_blocks)

    def get_cached_origin_blocks(
            self, request_id: str, num_computed_tokens: int,
            inner_blocks: list[int]) -> tuple[list[int], list[int]]:
        """
        Get cached outer blocks and their lengths for a given request.
        """
        result = self._cache_search_manager.find_cached_blocks(
            request_id, num_computed_tokens, inner_blocks,
            self._mapping_manager)

        return result.cached_outer_blocks, result.cached_lengths

    def get_blocks(self, request_id: str) -> torch.Tensor:
        """
        Get the tensor of outer block IDs for a given request.
        """
        block_ids = self._mapping_manager.get_request_blocks(request_id)
        return self._memory_pool_manager.get_tensor_for_blocks(block_ids)
