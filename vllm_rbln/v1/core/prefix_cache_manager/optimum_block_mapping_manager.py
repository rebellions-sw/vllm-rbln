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


class InnerBlockGroupManager:
    """
    The blocks with same block hashes are grouped together.
    """
    def __init__(self):
        self.groups: dict[str, set[int]] = {}
        self.inner_to_group_id: dict[int, int] = {}

    def register_inner_block(self, inner_block_id: int, same_inner_block_id: int) -> None:
        assert inner_block_id not in self.inner_to_group_id, \
            f"IB: {inner_block_id} already registered"
        assert inner_block_id not in self.groups[group_id], \
            f"IB: {inner_block_id} already in group {group_id}"
        if same_inner_block_id is None:
            group_id = self.create_group()
        else:
            group_id = self.get_group_id(same_inner_block_id)
        self.set_group_id(inner_block_id, group_id)

    def unregister_inner_block(self, inner_block_id: int) -> None:
        group_id = self.get_group_id(inner_block_id)
        self.groups[group_id].remove(inner_block_id)
        self.inner_to_group_id.pop(inner_block_id)
        if len(self.groups[group_id]) == 0:
            self.groups.pop(group_id)

    def set_group_id(self, inner_block_id: int, group_id: int) -> None:
        self.inner_to_group_id[inner_block_id] = group_id
        self.groups[group_id].add(inner_block_id)

    def get_group_id(self, inner_block_id: int) -> int:
        return self.inner_to_group_id.get(inner_block_id)
    
    def create_group(self) -> int:
        group_id = str(time.time())
        self.groups[group_id] = set[int]()
        return len(self.groups) - 1

    def get_group_ids(self, inner_block_ids: list[int]) -> list[int]:
        return [self.get_group_id(ib_id) for ib_id in inner_block_ids]


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
    Note:
        - _block_mappings, inner_to_outer, _request_to_outer_blocks
            are used for managing all block mappings.
    """

    def __init__(self):
        self._block_mappings: dict[int, BlockMapping] = {}
        self._inner_to_outer: dict[int, int] = {}
        self._request_to_outer_blocks: dict[str, list[int]] = {}
        # self._outer_to_cached_inner: dict[int, list[int]] = {}
        # self._cached_inner_to_outers: dict[int, list[int]] = {}

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
        # if inner_block_id in self._cached_inner_to_outers:
        #     # self._cached_inner_to_outers.pop(inner_block_id)
        #     ob_ids = list(self._outer_to_cached_inner.keys())
        #     for ob_id in ob_ids:
        #         cached_ibs = self._outer_to_cached_inner[ob_id]
        #         if inner_block_id in cached_ibs:
        #             cached_ibs.remove(inner_block_id)
        #             if len(cached_ibs) == 0:
        #                 self._outer_to_cached_inner.pop(ob_id, None)

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
        # cached_inner_block_ids = self.get_cached_inner_blocks_for_outer(
        #     outer_block_id)
        inner_block_ids = self.get_inner_blocks_for_outer(outer_block_id)
        logger.debug(
            "[PFX] [MAPPING-REMOVE] OB=%d | "
            "IB_COUNT=%d IB=%s",
            outer_block_id, len(inner_block_ids), inner_block_ids)
        # 1. Remove inner_block_id mapped to the removed outer_block_id
        for ib_id in inner_block_ids:
            self._inner_to_outer.pop(ib_id, None)
        # 2. Remove outer_block_id from cached_inner_to_outers mapping
        # The contents of ib_id cannot be accessed through
        # outer_block_id anymore.
        # for ib_id in cached_inner_block_ids:
        #     self._cached_inner_to_outers[ib_id].remove(outer_block_id)
        #     if len(self._cached_inner_to_outers[ib_id]) == 0:
        #         self._cached_inner_to_outers.pop(ib_id)

        # 3. Reset the outer_block_id to inner mapping
        mapping = self._block_mappings.pop(outer_block_id, None)
        # self._outer_to_cached_inner.pop(outer_block_id, None)

        return mapping

    def get_request_blocks(self, request_id: str) -> list[int]:
        """
        Return the list of outer block IDs associated with a request.
        """
        if not self.is_request_registered(request_id):
            raise ValueError(f"Request {request_id} is not registered")
        return self._request_to_outer_blocks[request_id].copy()

    def get_request_ids(self) -> list[str]:
        """
        Return the list of request IDs.
        """
        return list(self._request_to_outer_blocks.keys())

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

    def get_inner_blocks_for_request(self, request_id: str) -> list[int]:
        """
        Return the list of inner block IDs that map to a given request ID.
        """
        outer_blocks = self.get_request_blocks(request_id)
        return [self.get_inner_blocks_for_outer(ob_id) for ob_id in outer_blocks]

    # def get_cached_inner_blocks_for_outer(self,
    #                                       outer_block_id: int) -> list[int]:
    #     """
    #     Return the list of inner block IDs that are cached
    #     for a given outer block ID.
    #     """
    #     return self._outer_to_cached_inner.get(outer_block_id, [])

    # def get_outer_blocks_for_cached_inner(self,
    #                                       inner_block_id: int) -> list[int]:
    #     """
    #     Return the list of outer block IDs
    #     that map to a given cached inner block ID.
    #     """
    #     return self._cached_inner_to_outers.get(inner_block_id, [])
