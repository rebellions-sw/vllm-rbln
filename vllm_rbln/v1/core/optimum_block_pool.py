from vllm.v1.core.block_pool import BlockPool
from vllm.v1.core.kv_cache_utils import BlockHash, KVCacheBlock
from typing import Optional
from vllm.v1.core.kv_cache_utils import make_block_hash_with_group_id
from vllm.v1.core.kv_cache_utils import (BlockHash, BlockHashWithGroupId,
                                         ExternalBlockHash,
                                         FreeKVCacheBlockQueue, KVCacheBlock,
                                         get_block_hash,
                                         make_block_hash_with_group_id,
                                         maybe_convert_block_hash)
from vllm_rbln.logger import init_logger
logger = init_logger(__name__)
class RBLNBlockPool(BlockPool):

    def get_cached_block(
            self, block_hash: BlockHash,
            kv_cache_group_ids: list[int]) -> Optional[list[KVCacheBlock]]:
        """Get the cached block by the block hash for each group in 
        `kv_cache_group_ids`, or None if cache miss for any group.
        If there are duplicated blocks, we return the first block in the cache.

        Args:
            block_hash: The hash value of the block.
            kv_cache_group_ids: The ids of the KV cache groups.

        Returns:
            The cached blocks if exists, or None.
        """
        cached_blocks = []
        for group_id in kv_cache_group_ids:
            block_hash_with_group_id = make_block_hash_with_group_id(
                block_hash, group_id)
            cached_blocks_one_group = self.cached_block_hash_to_block.get(
                block_hash_with_group_id)
            if not cached_blocks_one_group:
                return None
            cached_blocks_one_group_values = cached_blocks_one_group.values()
            if len(cached_blocks_one_group_values) == 1:
                return None
            first_block = next(iter(cached_blocks_one_group_values))
            outer_block = []
            for block in cached_blocks_one_group_values:
                outer_block.append(block.block_id)
            logger.debug(f"Selected {first_block.block_id} | not_selected_blocks: {outer_block}")
            cached_blocks.append(first_block)
        return cached_blocks