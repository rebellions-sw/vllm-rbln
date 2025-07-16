from vllm.v1.core.kv_cache_manager import KVCacheManager
from vllm_rbln.core.kv_cache_coordinator_v1 import get_kv_cache_coordinator

from collections import defaultdict
from dataclasses import dataclass
from typing import Optional

from vllm.distributed.kv_events import KVCacheEvent
from vllm.logger import init_logger
from vllm.utils import sha256
from vllm.v1.core.kv_cache_utils import (BlockHash, KVCacheBlock,
                                         hash_request_tokens)
from vllm.v1.kv_cache_interface import KVCacheConfig
from vllm.v1.metrics.stats import PrefixCacheStats
from vllm.v1.request import Request, RequestStatus


class RBLNOptimumKVCacheManager(KVCacheManager):
    # _allocate_sequence에서는 중간에 BlockTable -> RBLNOptimumBlockTable로 바뀜
    # can_allocate에서는 중간에 if seq_group.is_encoder_decoder():만 빠짐
    # Allocate에서는 if seq_group.is_encoder_decoder():에서 block_table = self._allocate_sequence(encoder_seq) 빠짐
    pass

    # 호출하는 함수
    # get_block_ids
    # get_common_prefix_blocks
    # take_events
    # allocate_slots
    # free
    # get_computed_blocks
    # create_empty_block_list
    def __init__(
        self,
        kv_cache_config: KVCacheConfig,
        max_model_len: int,
        enable_caching: bool = True,
        caching_hash_algo: str = "builtin",
        use_eagle: bool = False,
        log_stats: bool = False,
        enable_kv_cache_events: bool = False,
    ) -> None:
        self.max_model_len = max_model_len

        self.enable_caching = enable_caching
        self.caching_hash_fn = sha256 if caching_hash_algo == "sha256" else hash
        self.use_eagle = use_eagle
        self.log_stats = log_stats
        # FIXME: make prefix cache stats conditional on log_stats
        self.prefix_cache_stats = PrefixCacheStats() if log_stats else None
        assert len(
            set(g.kv_cache_spec.block_size
                for g in kv_cache_config.kv_cache_groups)
        ) == 1, "Only one block size is supported for now"
        self.block_size = kv_cache_config.kv_cache_groups[
            0].kv_cache_spec.block_size

        self.coordinator = get_kv_cache_coordinator(
            kv_cache_config=kv_cache_config,
            max_model_len=self.max_model_len,
            use_eagle=self.use_eagle,
            enable_caching=enable_caching,
            caching_hash_fn=self.caching_hash_fn,
            enable_kv_cache_events=enable_kv_cache_events,
        )
        self.num_kv_cache_groups = len(kv_cache_config.kv_cache_groups)
        self.block_pool = self.coordinator.block_pool
        self.kv_cache_config = kv_cache_config

        # Mapping from request ID to kv block hashes.
        # This is to avoid recomputing the block hashes for each call of
        # `get_computed_blocks` or `allocate_slots`.
        self.req_to_block_hashes: defaultdict[
            str, list[BlockHash]] = defaultdict(list)