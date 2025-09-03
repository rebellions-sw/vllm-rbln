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

from vllm.v1.core.kv_cache_utils import BlockHash, KVCacheBlock

PREFIX_CACHING_BLOCK_SIZE = 128


class RBLNFullAttentionManager(FullAttentionManager):

    @classmethod
    def find_longest_cache_hit(
        cls,
        block_hashes: list[BlockHash],
        max_length: int,
        kv_cache_group_ids: list[int],
        block_pool: BlockPool,
        kv_cache_spec: KVCacheSpec,
        use_eagle: bool,
    ) -> tuple[list[KVCacheBlock], ...]:
        assert isinstance(kv_cache_spec, FullAttentionSpec), (
            "FullAttentionManager can only be used for full attention groups")
        computed_blocks: tuple[list[KVCacheBlock], ...] = tuple(
            [] for _ in range(len(kv_cache_group_ids)))
        max_num_blocks = max_length // PREFIX_CACHING_BLOCK_SIZE
        for i, block_hash in zip(range(max_num_blocks), block_hashes):
            # block_hashes is a chain of block hashes. If a block hash is not
            # in the cached_block_hash_to_id, the following block hashes are
            # not computed yet for sure.
            if cached_block := block_pool.get_cached_block(
                    block_hash, kv_cache_group_ids):
                for computed, cached in zip(computed_blocks, cached_block):
                    computed.append(cached)
            else:
                break
        if use_eagle and computed_blocks[0]:
            for computed in computed_blocks:
                computed.pop()
        return computed_blocks

    def get_inner_blocks(
        self,
        
    )