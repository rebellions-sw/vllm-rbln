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
from typing import Callable

from vllm.v1.core.block_pool import BlockPool
from vllm.v1.core.kv_cache_coordinator import (KVCacheCoordinator,
                                               UnitaryKVCacheCoordinator)
from vllm.v1.core.kv_cache_utils import BlockHash, KVCacheBlock
from vllm.v1.core.single_type_kv_cache_manager import (
    get_manager_for_kv_cache_spec)
from vllm.v1.kv_cache_interface import KVCacheConfig
from vllm.v1.request import Request

from vllm_rbln.v1.core.single_type_kv_cache_manager import (
    RBLNFullAttentionManager)

PREFIX_CACHING_BLOCK_SIZE = 128


class RBLNUnitaryKVCacheCoordinator(UnitaryKVCacheCoordinator):

    def __init__(
        self,
        kv_cache_config: KVCacheConfig,
        max_model_len: int,
        use_eagle: bool,
        enable_caching: bool,
        caching_hash_fn: Callable,
        enable_kv_cache_events: bool,
    ):
        self.kv_cache_config = kv_cache_config
        self.block_size = kv_cache_config.kv_cache_groups[
            0].kv_cache_spec.block_size
        self.max_model_len = max_model_len

        self.block_pool = RBLNBlockPool(kv_cache_config.num_blocks, self.enable_caching,
                                    enable_kv_cache_events)
        self.use_eagle = use_eagle
        self.single_type_managers = tuple(
            RBLNFullAttentionManager(
                kv_cache_spec=kv_cache_group.kv_cache_spec,
                block_pool=self.block_pool,
                kv_cache_group_id=i,
                caching_hash_fn=caching_hash_fn,
            ) for i, kv_cache_group in enumerate(
                self.kv_cache_config.kv_cache_groups))

    def find_longest_cache_hit(
        self,
        block_hashes: list[BlockHash],
        max_cache_hit_length: int,
    ) -> tuple[tuple[list[KVCacheBlock], ...], int]:
        """
        Get the longest blocks that matched with already cached blocks
        and the offset of the last cached block.

        Args:
            block_hashes: The block hashes of the request.
            max_cache_hit_length: The maximum length of tokens
                that are available to hit the cache.

        Returns:
            - A list of cached blocks for each kv cache group
            - The number of new tokens that are already computed
                (prefix caching)
        """
        # max_cache_hit_length = request.num_tokens - 1 고정
        hit_blocks, num_sub_blocks = self.single_type_managers[0].custom_find_longest_cache_hit(
            block_hashes=block_hashes,
            max_length=max_cache_hit_length,
            kv_cache_group_ids=[0],
            block_pool=self.prefix_caching_block_pool,
            kv_cache_spec=self.kv_cache_spec,
            use_eagle=self.use_eagle,
        )

        return hit_blocks, num_sub_blocks * PREFIX_CACHING_BLOCK_SIZE

def get_kv_cache_coordinator(
        kv_cache_config: KVCacheConfig, max_model_len: int, use_eagle: bool,
        enable_caching: bool, caching_hash_fn: Callable,
        enable_kv_cache_events: bool) -> KVCacheCoordinator:

    assert len(kv_cache_config.kv_cache_groups
               ) == 1, "vLLM RBLN requires only one type of kv_cache_group."
    return RBLNUnitaryKVCacheCoordinator(kv_cache_config, max_model_len,
                                         use_eagle, enable_caching,
                                         caching_hash_fn,
                                         enable_kv_cache_events)
