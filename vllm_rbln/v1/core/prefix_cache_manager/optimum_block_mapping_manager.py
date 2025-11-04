# Copyright 2025 Rebellions Inc. All rights reserved.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at:

#     http://www.apache.org/licenses/LICENSE-2.0

from dataclasses import dataclass
from typing import Optional

from vllm_rbln.logger import init_logger

logger = init_logger(__name__)


class RBLNBlock:

    def __init__(self, block_id: int):
        self.block_id = block_id

    def __repr__(self) -> str:
        return f"RBLNBlock(id={self.block_id})"


@dataclass
class BlockMapping:
    outer_block_id: int
    inner_block_ids: list[int]
    request_id: Optional[str] = None
    is_active: bool = True


@dataclass
class OuterBlockInfo:
    outer_block_id: int
    num_cached_ibs: int


class BlockMappingManager:
    """
    Manage mappings between outer blocks and inner blocks.
    Also manage mappings between requests and their allocated outer blocks.
    """

    def __init__(self):
        self._outer_to_inner: dict[int, BlockMapping] = {}
        self._inner_to_outer: dict[int, int] = {}
        self._request_mappings: dict[str, list[int]] = {}
        self._matched: dict[int, list[OuterBlockInfo]] = {}

    def is_request_registered(self, request_id: str) -> bool:
        """
        Check if a request ID is registered in the manager.
        """
        return request_id in self._request_mappings

    def is_inner_block_mapped(self, inner_block_id: int) -> bool:
        """
        Check if an inner block ID is mapped in the manager.
        """
        return inner_block_id in self._inner_to_outer

    def add_new_inner_to_outer(self, inner_block_id: int,
                               outer_block_id: int) -> None:
        """
        Add a new mapping from an inner block ID to an outer block ID.
        """
        # if inner_block_id not in self._inner_to_outer:
        #     self._inner_to_outer[inner_block_id] = []
        self._inner_to_outer[inner_block_id] = outer_block_id

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
            self.add_new_inner_to_outer(ib_id, outer_block.block_id)

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
                    del self._inner_to_outer[ib_id]
                if ib_id in self._matched:
                    del self._matched[ib_id]

        return mapping

    def get_request_blocks(self, request_id: str) -> list[int]:
        """
        Return the list of outer block IDs associated with a request.
        """
        return self._request_mappings.get(request_id, []).copy()

    def remove_request(self, request_id: str) -> list[int]:
        """
        Remove all mappings associated with a request
        and return the outer block IDs.
        """
        return self._request_mappings.pop(request_id, [])

    def get_mapping(self, outer_block_id: int) -> Optional[BlockMapping]:
        """
        Return the mapping for a given outer block ID.
        """
        return self._outer_to_inner.get(outer_block_id)

    def get_outer_block_for_inner(self, inner_block_id: int) -> int:
        """
        Return the outer block ID that maps to a given inner block ID.
        """
        return self._inner_to_outer.get(inner_block_id)

    def get_inner_blocks_for_outer(self, outer_block_id: int) -> list[int]:
        """
        Return the list of inner block IDs that map to a given outer block ID.
        """
        mapping = self._outer_to_inner.get(outer_block_id)
        return mapping.inner_block_ids if mapping else []

    def get_inactive_mappings(self) -> list[BlockMapping]:
        """
        Return a list of inactive mappings.
        """
        return [
            mapping for mapping in self._outer_to_inner.values()
            if not mapping.is_active
        ]

    def match_cached_blocks(self, cached_ib: list[int],
                            outer_block_ids: list[int],
                            block_ratio: int) -> None:
        """
        Record matched outer block id between cached inner blocks.
        """
        for outer_block_id in outer_block_ids:
            start_ib_idx = outer_block_id * block_ratio
            end_ib_idx = min(start_ib_idx + block_ratio, len(cached_ib))
            cur_ib_segment = cached_ib[start_ib_idx:end_ib_idx]
            first_unique_ib = cur_ib_segment[0]
            if first_unique_ib not in self._matched:
                self._matched[first_unique_ib] = []
            outer_block_info = OuterBlockInfo(outer_block_id,
                                              len(cur_ib_segment))
            self._matched[first_unique_ib].append(outer_block_info)
            logger.debug(
                "[PFX] Matched outer block id %d "
                "for cached inner block segment %s", outer_block_id,
                cur_ib_segment)

    def get_outer_block_for_copied_inner(
            self, cached_ib_segment: list[int]) -> tuple[int, int]:
        """
        Return the matched outer block ID
        for a given cached inner block segment.
        """
        matched_obs = self._matched.get(cached_ib_segment[0])
        if matched_obs is not None:
            alive_obs = [
                ob for ob in matched_obs
                if ob.outer_block_id in self._outer_to_inner
            ]
            if len(alive_obs) > 0:
                outer_block_id, num_cached_ibs = alive_obs[
                    0].outer_block_id, alive_obs[0].num_cached_ibs
                if num_cached_ibs >= len(cached_ib_segment):
                    return outer_block_id, num_cached_ibs
        return -1, 0
