from vllm_rbln.core.block_pool_v1 import RBLNOptimumBlockPool

from abc import ABC, abstractmethod
from typing import Callable, Optional

from vllm.v1.core.block_pool import BlockPool
from vllm.v1.core.kv_cache_utils import BlockHash, KVCacheBlock
from vllm.v1.core.single_type_kv_cache_manager import (
    FullAttentionManager, get_manager_for_kv_cache_spec)
from vllm.v1.kv_cache_interface import FullAttentionSpec, KVCacheConfig
from vllm.v1.request import Request
from vllm.v1.core.kv_cache_coordinator import KVCacheCoordinator, UnitaryKVCacheCoordinator, HybridKVCacheCoordinator

class RBLNOptimumKVCacheCoordinator(KVCacheCoordinator):
    """
    Coordinate the KV cache of different KV cache groups.
    """

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
        self.max_model_len = max_model_len
        self.enable_caching = enable_caching

        self.block_pool = RBLNOptimumBlockPool(kv_cache_config.num_blocks, enable_caching,
                                    enable_kv_cache_events)

        # Needs special handling for find_longest_cache_hit if eagle is enabled
        self.use_eagle = use_eagle
        self.single_type_managers = tuple(
            get_manager_for_kv_cache_spec(
                kv_cache_spec=kv_cache_group.kv_cache_spec,
                block_pool=self.block_pool,
                kv_cache_group_id=i,
                caching_hash_fn=caching_hash_fn,
            ) for i, kv_cache_group in enumerate(
                self.kv_cache_config.kv_cache_groups))


class RBLNOptimumUnitaryKVCacheCoordinator(RBLNOptimumKVCacheCoordinator, UnitaryKVCacheCoordinator):
    pass

class RBLNOptimumHybridKVCacheCoordinator(RBLNOptimumKVCacheCoordinator, HybridKVCacheCoordinator):
    pass


def get_kv_cache_coordinator(
        kv_cache_config: KVCacheConfig, max_model_len: int, use_eagle: bool,
        enable_caching: bool, caching_hash_fn: Callable,
        enable_kv_cache_events: bool) -> RBLNOptimumKVCacheCoordinator:
    if len(kv_cache_config.kv_cache_groups) == 1:
        return RBLNOptimumUnitaryKVCacheCoordinator(kv_cache_config, max_model_len,
                                         use_eagle, enable_caching,
                                         caching_hash_fn,
                                         enable_kv_cache_events)
    return RBLNOptimumHybridKVCacheCoordinator(kv_cache_config, max_model_len, use_eagle,
                                    enable_caching, caching_hash_fn,
                                    enable_kv_cache_events)