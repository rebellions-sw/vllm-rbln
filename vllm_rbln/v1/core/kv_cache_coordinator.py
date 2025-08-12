from vllm.v1.kv_cache_interface import (FullAttentionSpec, KVCacheConfig,
                                        KVCacheSpec)
from vllm.v1.core.kv_cache_coordinator import KVCacheCoordinator
from vllm.v1.core.block_pool import BlockPool
from vllm_rbln.v1.core.single_type_kv_cache_manager import get_manager_for_kv_cache_spec
from typing import Callable
from vllm.v1.core.kv_cache_utils import BlockHash, KVCacheBlock
class RBLNKVCacheCoordinator(KVCacheCoordinator):

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

        self.block_pool = BlockPool(kv_cache_config.num_blocks, enable_caching,
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

class RBLNUnitaryKVCacheCoordinator(RBLNKVCacheCoordinator):
    """
    KV cache coordinator for models with only one KV cache group. This is the
    case for models with only one KV cache type, e.g., all attention layers use
    full attention or all attention layers use sliding window attention.
    """

    def __init__(self, kv_cache_config: KVCacheConfig, max_model_len: int,
                 use_eagle: bool, enable_caching: bool,
                 caching_hash_fn: Callable, enable_kv_cache_events: bool):
        super().__init__(kv_cache_config, max_model_len, use_eagle,
                         enable_caching, caching_hash_fn,
                         enable_kv_cache_events)
        self.kv_cache_spec = self.kv_cache_config.kv_cache_groups[
            0].kv_cache_spec
        self.block_size = self.kv_cache_spec.block_size
        assert len(self.kv_cache_config.kv_cache_groups) == 1, (
            "UnitaryKVCacheCoordinator assumes only one kv cache group")

    def find_longest_cache_hit(
        self,
        block_hashes: list[BlockHash],
        max_cache_hit_length: int,
    ) -> tuple[tuple[list[KVCacheBlock], ...], int]:
        hit_blocks = self.single_type_managers[0].find_longest_cache_hit(
            block_hashes=block_hashes,
            max_length=max_cache_hit_length,
            kv_cache_group_ids=[0],
            block_pool=self.block_pool,
            kv_cache_spec=self.kv_cache_spec,
            use_eagle=self.use_eagle,
        )
        return hit_blocks, len(hit_blocks[0]) * self.block_size


def get_kv_cache_coordinator(
        kv_cache_config: KVCacheConfig, max_model_len: int, use_eagle: bool,
        enable_caching: bool, caching_hash_fn: Callable,
        enable_kv_cache_events: bool) -> KVCacheCoordinator:
    if len(kv_cache_config.kv_cache_groups) == 1:
        return RBLNUnitaryKVCacheCoordinator(kv_cache_config, max_model_len,
                                         use_eagle, enable_caching,
                                         caching_hash_fn,
                                         enable_kv_cache_events)
    else:
        return RuntimeError("Multiple KV cache groups are not supported yet.")