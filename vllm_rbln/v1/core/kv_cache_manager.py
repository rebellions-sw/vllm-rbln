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

from vllm.v1.core.kv_cache_manager import KVCacheManager

from vllm_rbln.v1.core.kv_cache_coordinator import get_kv_cache_coordinator

PREFIX_CACHING_BLOCK_SIZE = 128


class RBLNKVCacheManager(KVCacheManager):

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
        self.prefix_caching_block_pool = self.coordinator.prefix_caching_block_pool
        self.kv_cache_config = kv_cache_config

        # Mapping from request ID to kv block hashes.
        # This is to avoid recomputing the block hashes for each call of
        # `get_computed_blocks` or `allocate_slots`.
        self.req_to_block_hashes: defaultdict[
            str, list[BlockHash]] = defaultdict(list)

    def get_computed_blocks(self,
                            request: Request) -> tuple[KVCacheBlocks, int]:
        """Get the computed (cached) blocks for the request.
        Note that the computed blocks must be full.

        Args:
            request: The request to get the computed blocks.

        Returns:
            A tuple containing:
                - A list of blocks that are computed for the request.
                - The number of computed tokens.
        """
        # Prefix caching is disabled or
        # When the request requires prompt logprobs, we skip prefix caching.
        if (not self.enable_caching
                or request.sampling_params.prompt_logprobs is not None):
            return self.create_empty_block_list(), 0

        # The block hashes for the request may already be computed
        # if the scheduler has tried to schedule the request before.
        block_hashes = self.req_to_block_hashes[request.request_id]
        if not block_hashes:
            # NOTE(eunji) MODIFIED
            block_hashes = hash_request_tokens(self.caching_hash_fn,
                                               PREFIX_CACHING_BLOCK_SIZE,
                                               request)
            self.req_to_block_hashes[request.request_id] = block_hashes

        if self.log_stats:
            assert self.prefix_cache_stats is not None
            self.prefix_cache_stats.requests += 1

        # NOTE: When all tokens hit the cache, we must recompute the last token
        # to obtain logits. Thus, set max_cache_hit_length to prompt_length - 1.
        # This can trigger recomputation of an entire block, rather than just
        # the single last token, because allocate_slots() requires
        # num_computed_tokens to be block-size aligned. Removing this limitation
        # could slightly improve performance in the future.
        max_cache_hit_length = request.num_tokens - 1
        computed_blocks, num_new_computed_tokens = (
            self.coordinator.find_longest_cache_hit(block_hashes,
                                                    max_cache_hit_length))

        if self.log_stats:
            assert self.prefix_cache_stats is not None
            self.prefix_cache_stats.queries += request.num_tokens
            self.prefix_cache_stats.hits += num_new_computed_tokens

        return KVCacheBlocks(computed_blocks), num_new_computed_tokens
