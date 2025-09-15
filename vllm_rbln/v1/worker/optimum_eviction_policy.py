from collections import OrderedDict
import time
from vllm_rbln.logger import init_logger
from vllm_rbln.v1.worker.optimum_block_mapping_manager import BlockMappingManager

logger = init_logger(__name__)

class RREvictionPolicy:
    """
    Simple eviction policy to select blocks for eviction.
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

        if len(evicted_blocks) < count:
            logger.warning(
                f"Could not find enough inactive blocks for eviction. "
                f"Requested: {count}, Found: {len(evicted_blocks)}")

        return evicted_blocks


class LRUEvictionPolicy(RREvictionPolicy):
    """
    LRU (Least Recently Used) eviction policy implementation.
    """
    
    def __init__(self):
        self._access_order: OrderedDict[int, float] = OrderedDict()
    
    def touch(self, block_id: int) -> None:
        self._access_order[block_id] = time.time()
        # Move to the end to mark as most recently used
        self._access_order.move_to_end(block_id)
    
    def select_blocks_for_eviction(self, mapping_manager, count: int) -> list[int]:
        inactive_mappings = mapping_manager.get_inactive_mappings()
        inactive_block_ids = [m.outer_block_id for m in inactive_mappings]
        
        # Sort the blocks in LRU order (oldest first)
        evictable_blocks = [
            block_id for block_id in self._access_order.keys()
            if block_id in inactive_block_ids
        ]
        
        selected = evictable_blocks[:count]
        
        # Remove the selected blocks from the record
        for block_id in selected:
            self._access_order.pop(block_id, None)
        
        return selected

