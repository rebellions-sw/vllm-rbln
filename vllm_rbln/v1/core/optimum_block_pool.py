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

from typing import Optional

from vllm.v1.core.block_pool import BlockPool
from vllm.v1.core.kv_cache_utils import (BlockHash, KVCacheBlock,
                                         make_block_hash_with_group_id)

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
            # NOTE(eunji.lee)
            # Exclude blocks allocated by the current request itself
            if len(cached_blocks_one_group_values) <= 1:
                return None
            iterator = reversed(cached_blocks_one_group_values)
            _ = next(iterator)
            second_last_block = next(iterator)
            cached_blocks.append(second_last_block)
        return cached_blocks
