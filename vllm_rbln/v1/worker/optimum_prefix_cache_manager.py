# Copyright 2025 Rebellions Inc. All rights reserved.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at:

#     http://www.apache.org/licenses/LICENSE-2.0

import time
from abc import ABC, abstractmethod
from vllm_rbln.v1.worker.optimum_eviction_policy import (LRUEvictionPolicy,
                                                 RREvictionPolicy)
from vllm_rbln.v1.worker.optimum_block_mapping_manager import BlockMappingManager, RBLNBlock
from collections import OrderedDict, deque
from dataclasses import dataclass
from typing import Optional

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
class CacheSearchResult:
    cached_outer_blocks: list[int]
    cached_lengths: list[int]
    source_request_id: Optional[str] = None

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
        # 실제 할당된 블록 객체들을 추적
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

    def get_allocated_block(self, block_id: int) -> Optional[RBLNBlock]:
        """할당된 블록 객체 반환"""
        return self._allocated_blocks.get(block_id)

    def get_free_count(self) -> int:
        return len(self._free_blocks)

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
        self._eviction_policy = RREvictionPolicy()

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
            # New outer block needed
            self._allocate_new_blocks(request_id, 1, inner_blocks)
        else:
            # Append to the last outer block
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
        block_ids = []
        for i, block in enumerate(new_blocks):
            start_idx = i * self._config.block_ratio
            end_idx = min((i + 1) * self._config.block_ratio,
                          len(inner_blocks))
            block_inner_blocks = inner_blocks[start_idx:end_idx]

            self._mapping_manager.create_mapping(block, block_inner_blocks,
                                                 request_id)
            if isinstance(self._eviction_policy, LRUEvictionPolicy):
                self._eviction_policy.register_block(block.block_id)
            block_ids.append(block.block_id)

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

        # 충분한 블록을 확보할 수 없는 경우 예외 발생
        if len(blocks_to_evict) < evict_count:
            raise RuntimeError(
                f"Cannot evict enough blocks. Need {evict_count}, "
                f"can evict {len(blocks_to_evict)}")

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
                if isinstance(self._eviction_policy, LRUEvictionPolicy):
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

        if result.has_cache_hit and isinstance(self._eviction_policy,
                                               LRUEvictionPolicy):
            for ob_id in result.cached_outer_blocks:
                self._eviction_policy.touch(ob_id)

        return result.cached_outer_blocks, result.cached_lengths

    def get_blocks(self, request_id: str) -> torch.Tensor:
        """
        Get the tensor of outer block IDs for a given request.
        """
        if request_id not in self._mapping_manager._request_mappings:
            logger.warning("Request %s not found in mappings", request_id)
            return self._memory_pool_manager.get_tensor_for_blocks([])

        block_ids = self._mapping_manager.get_request_blocks(request_id)
        return self._memory_pool_manager.get_tensor_for_blocks(block_ids)
