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
from collections import OrderedDict

from vllm_rbln.logger import init_logger
from vllm_rbln.v1.worker.optimum_block_mapping_manager import (
    BlockMappingManager)

logger = init_logger(__name__)


class SimpleEvictionPolicy:
    """
    Simple eviction policy to select blocks for eviction.
    """

    def register_block(self, block_id: int) -> None:
        """Register a new block (called when block is first allocated)"""
        pass

    def unregister_block(self, block_id: int) -> None:
        """Unregister a block (called when block is deallocated)"""
        pass

    def select_blocks_for_eviction(self, mapping_manager: BlockMappingManager,
                                   count: int) -> list[int]:
        # Select blocks for eviction.
        inactive_mappings = mapping_manager.get_inactive_mappings()
        evicted_blocks = []

        for mapping in inactive_mappings:
            if len(evicted_blocks) >= count:
                break
            evicted_blocks.append(mapping.outer_block_id)

        if len(evicted_blocks) < count:
            logger.warning(
                "Could not find enough inactive blocks for eviction. "
                "Requested: %d, Found: %d", count, len(evicted_blocks))

        return evicted_blocks[:count]


class FIFOEvictionPolicy(SimpleEvictionPolicy):
    """
    FIFO (First In First Out) eviction policy implementation.
    """

    def __init__(self):
        self._allocation_order: OrderedDict[int, float] = OrderedDict()

    def register_block(self, block_id: int) -> None:
        assert block_id not in self._allocation_order
        self._allocation_order[block_id] = True

    def unregister_block(self, block_id: int) -> None:
        self._allocation_order.pop(block_id, None)

    def select_blocks_for_eviction(self, mapping_manager,
                                   count: int) -> list[int]:
        # TODO If the cached block is evicted, we should also evict its mapping
        # How about exclude the cached blocks from eviction?
        # AS-IS: Eviction -> Cache check -> Allocation
        # TO-DO: Cache check -> Eviction -> Allocation (more complicated)
        inactive_mappings = mapping_manager.get_inactive_mappings()
        inactive_block_ids = [m.outer_block_id for m in inactive_mappings]

        if not inactive_block_ids:
            logger.warning("No inactive blocks available for eviction")
            return []

        evictable_blocks = [
            block_id for block_id in self._allocation_order
            if block_id in inactive_block_ids
        ]

        selected = evictable_blocks[:count]

        for block_id in selected:
            self._allocation_order.pop(block_id, None)

        if len(selected) < count:
            logger.warning(
                "Could not find enough inactive blocks for eviction. "
                "Requested: %d, Found: %d", count, len(selected))

        return selected


class LRUEvictionPolicy(SimpleEvictionPolicy):
    """
    LRU (Least Recently Used) eviction policy implementation.
    """

    def __init__(self):
        self._access_order: OrderedDict[int, float] = OrderedDict()

    def touch(self, block_id: int) -> None:
        """Mark a block as recently accessed"""
        # Move to the end to mark as most recently used
        self._access_order.move_to_end(block_id)

    def register_block(self, block_id: int) -> None:
        assert block_id not in self._access_order
        self._access_order[block_id] = True

    def unregister_block(self, block_id: int) -> None:
        self._access_order.pop(block_id, None)

    def select_blocks_for_eviction(self, mapping_manager,
                                   count: int) -> list[int]:
        inactive_mappings = mapping_manager.get_inactive_mappings()
        inactive_block_ids = [m.outer_block_id for m in inactive_mappings]

        if not inactive_block_ids:
            logger.warning("No inactive blocks available for eviction")
            return []

        untouched_blocks = [
            block_id for block_id in inactive_block_ids
            if block_id not in self._access_order
        ]

        touched_blocks = [
            block_id for block_id in self._access_order
            if block_id in inactive_block_ids
        ]

        evictable_blocks = untouched_blocks + touched_blocks
        selected = evictable_blocks[:count]

        for block_id in selected:
            self._access_order.pop(block_id, None)

        if len(selected) < count:
            logger.warning(
                "Could not find enough inactive blocks for eviction. "
                "Requested: %d, Found: %d", count, len(selected))

        return selected
