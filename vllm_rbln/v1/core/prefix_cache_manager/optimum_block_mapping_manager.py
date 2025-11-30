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


class BlockMappingManager:
    """
    Manage mappings between outer blocks and inner blocks.
    Also manage mappings between requests and their allocated outer blocks.
    Args:
        _block_mappings: dict[int, BlockMapping]
            Mapping from outer block ID to BlockMapping.
        _inner_to_outer: dict[int, int]
            Mapping from inner block ID to outer block ID.
        _request_to_outer_blocks: dict[str, list[int]]
            Mapping from request ID to list of outer block IDs.
        _outer_to_cached_inner: dict[int, list[int]]
            Mapping from outer block ID to list of cached inner block IDs.
        _cached_inner_to_outers: dict[int, list[int]]
            Mapping from cached inner block ID to list of outer block IDs.
    Note:
        - _block_mappings, inner_to_outer, _request_to_outer_blocks
            are used for managing all block mappings.
        - _outer_to_cached_inner, _cached_inner_to_outers
            are used for managing cached blocks.
    """

    def __init__(self):
        self._block_mappings: dict[int, BlockMapping] = {}
        self._inner_to_outer: dict[int, int] = {}
        self._request_to_outer_blocks: dict[str, list[int]] = {}
        self._outer_to_cached_inner: dict[int, list[int]] = {}
        self._cached_inner_to_outers: dict[int, list[int]] = {}

    def is_request_registered(self, request_id: str) -> bool:
        """
        Check if a request ID is registered in the manager.
        """
        return request_id in self._request_to_outer_blocks

    def is_inner_block_mapped(self, inner_block_id: int) -> bool:
        """
        Check if an inner block ID is mapped in the manager.
        """
        return inner_block_id in self._inner_to_outer

    def add_new_inner_to_outer(self, inner_block_id: int,
                               outer_block_id: int) -> None:
        """
        Add a new mapping from an inner block ID to an outer block ID.
        And remove previous caching history of newly allocated
        inner blocks if exist (Lazy update).
        """
        if inner_block_id in self._cached_inner_to_outers:
            self._cached_inner_to_outers.pop(inner_block_id)
            ob_ids = list(self._outer_to_cached_inner.keys())
            for ob_id in ob_ids:
                cached_ibs = self._outer_to_cached_inner[ob_id]
                if inner_block_id in cached_ibs:
                    cached_ibs.remove(inner_block_id)
                    if len(cached_ibs) == 0:
                        self._outer_to_cached_inner.pop(ob_id, None)

        self._inner_to_outer[inner_block_id] = outer_block_id

    def create_mapping(self, outer_block: RBLNBlock, inner_blocks: list[int],
                       request_id: str) -> None:
        """
        Create a new block mapping.
        """
        mapping = BlockMapping(outer_block_id=outer_block.block_id,
                               inner_block_ids=inner_blocks.copy(),
                               request_id=request_id)
        self._block_mappings[outer_block.block_id] = mapping

        # Update Inner to outer mapping
        for ib_id in inner_blocks:
            self.add_new_inner_to_outer(ib_id, outer_block.block_id)

        # Update Request mapping
        if request_id not in self._request_to_outer_blocks:
            self._request_to_outer_blocks[request_id] = []
        self._request_to_outer_blocks[request_id].append(outer_block.block_id)

    def remove_mapping(self, outer_block_id: int) -> Optional[BlockMapping]:
        """
        Remove a mapping by outer block ID and return the removed mapping.
        """
        cached_inner_block_ids = self.get_cached_inner_blocks_for_outer(
            outer_block_id)
        inner_block_ids = self.get_inner_blocks_for_outer(outer_block_id)
        logger.debug(
            "[PFX] [MAPPING-REMOVE] OB=%d | "
            "IB_COUNT=%d IB=%s | "
            "CACHED_IB_COUNT=%d %s",
            outer_block_id, len(inner_block_ids), inner_block_ids,
            len(cached_inner_block_ids), cached_inner_block_ids)
        # 1. Remove inner_block_id mapped to the removed outer_block_id
        for ib_id in inner_block_ids:
            self._inner_to_outer.pop(ib_id, None)
        # 2. Remove outer_block_id from cached_inner_to_outers mapping
        # The contents of ib_id cannot be accessed through
        # outer_block_id anymore.
        for ib_id in cached_inner_block_ids:
            self._cached_inner_to_outers[ib_id].remove(outer_block_id)

        # 3. Reset the outer_block_id to inner mapping
        mapping = self._block_mappings.pop(outer_block_id, None)
        self._outer_to_cached_inner.pop(outer_block_id, None)

        return mapping

    def get_request_blocks(self, request_id: str) -> list[int]:
        """
        Return the list of outer block IDs associated with a request.
        """
        if not self.is_request_registered(request_id):
            raise ValueError(f"Request {request_id} is not registered")
        return self._request_to_outer_blocks[request_id].copy()

    def remove_request(self, request_id: str) -> list[int]:
        """
        Remove all mappings associated with a request
        and return the outer block IDs.
        """
        return self._request_to_outer_blocks.pop(request_id, [])

    def get_mapping(self, outer_block_id: int) -> Optional[BlockMapping]:
        """
        Return the mapping for a given outer block ID.
        """
        return self._block_mappings.get(outer_block_id)

    def get_outer_block_for_inner(self, inner_block_id: int) -> int:
        """
        Return the outer block ID that maps to a given inner block ID.
        """
        return self._inner_to_outer.get(inner_block_id)

    def get_inner_blocks_for_outer(self, outer_block_id: int) -> list[int]:
        """
        Return the list of inner block IDs that map to a given outer block ID.
        """
        mapping = self._block_mappings.get(outer_block_id)
        return mapping.inner_block_ids if mapping else []

    def get_inactive_mappings(self) -> list[BlockMapping]:
        """
        Return a list of inactive mappings.
        """
        return [
            mapping for mapping in self._block_mappings.values()
            if not mapping.is_active
        ]

    def get_cached_inner_blocks_for_outer(self,
                                          outer_block_id: int) -> list[int]:
        """
        Return the list of inner block IDs that are cached
        for a given outer block ID.
        """
        return self._outer_to_cached_inner.get(outer_block_id, [])

    def get_outer_blocks_for_cached_inner(self,
                                          inner_block_id: int) -> list[int]:
        """
        Return the list of outer block IDs
        that map to a given cached inner block ID.
        """
        return self._cached_inner_to_outers.get(inner_block_id, [])

    def set_cached_blocks(self, inner_blocks: list[int],
                          outer_block_ids: list[int],
                          block_ratio: int) -> None:
        """
        Set the cached blocks by mapping inner blocks to outer blocks.
        inner_blocks: list[int]
            List of inner block IDs that are cached
            or newly allocated inner block IDs.
        outer_block_ids: list[int]
            List of outer block IDs.
        """
        if len(inner_blocks) == 0:
            return

        for cur_outer_block_idx, start_ib_idx in enumerate(
                range(0, len(inner_blocks), block_ratio)):
            end_ib_idx = min(start_ib_idx + block_ratio, len(inner_blocks))
            cur_ib_segment = inner_blocks[start_ib_idx:end_ib_idx]
            cur_outer_block_id = outer_block_ids[cur_outer_block_idx]
            if self._outer_to_cached_inner.get(cur_outer_block_id) is None:
                # First segment
                self._outer_to_cached_inner[
                    cur_outer_block_id] = cur_ib_segment
            else:
                # Second segment
                self._outer_to_cached_inner[cur_outer_block_id].extend(
                    cur_ib_segment)

            for ib_id in cur_ib_segment:
                if ib_id not in self._cached_inner_to_outers:
                    self._cached_inner_to_outers[ib_id] = []
                self._cached_inner_to_outers[ib_id].append(cur_outer_block_id)

    def get_longest_matched_block(self, cached_ib_segment: list[int],
                                  skip_blocks: set[int]) -> tuple[int, int]:
        """
        Given a segment of cached inner block IDs,
        return the outer block ID that has the longest matching prefix
        with the cached inner block segment.
        If no match is found, return tuple[-1, 0].
        """
        # Find the outer block IDs
        # that match the first block of cached inner block segment
        matched_obs = self._cached_inner_to_outers.get(cached_ib_segment[0])
        logger.debug(
            "[PFX] [MAPPING-SEARCH] QUERY_IB=%d | "
            "SEGMENT_SIZE=%d SEGMENT=%s | "
            "CANDIDATE_OBS=%s",
            cached_ib_segment[0] if cached_ib_segment else -1,
            len(cached_ib_segment), cached_ib_segment,
            matched_obs if matched_obs else [])
        final_outer_block_id = -1
        final_num_ibs = 0
        if matched_obs is not None:
            alive_obs = [
                ob for ob in matched_obs if ob in self._block_mappings
            ]
            # TODO It is not required. But it is a safety check.
            assert len(matched_obs) == len(alive_obs)

            alive_obs = [ob for ob in alive_obs if ob not in skip_blocks]
            for outer_block_id in alive_obs:
                cached_ibs = self._outer_to_cached_inner[outer_block_id]
                prefix_ibs = self._get_common_prefix(cached_ibs,
                                                     cached_ib_segment)
                cache_hit_size = len(prefix_ibs)
                if cache_hit_size > final_num_ibs:
                    final_outer_block_id = outer_block_id
                    final_num_ibs = cache_hit_size
        return final_outer_block_id, final_num_ibs

    def _get_common_prefix(self, arr1: list[int],
                           arr2: list[int]) -> list[int]:
        """
        Return the common prefix between two lists of integers.
        """
        common_prefix = []
        min_length = min(len(arr1), len(arr2))
        for i in range(min_length):
            if arr1[i] == arr2[i]:
                common_prefix.append(arr1[i])
            else:
                break
        return common_prefix
