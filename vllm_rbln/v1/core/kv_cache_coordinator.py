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

        self.num_prefix_cache_blocks = kv_cache_config.num_blocks * (
            self.block_size // PREFIX_CACHING_BLOCK_SIZE)
        # Set enable_caching = False statically
        self.block_pool = BlockPool(kv_cache_config.num_blocks, False,
                                    enable_kv_cache_events)

        self.prefix_caching_block_pool = BlockPool(
            self.num_prefix_cache_blocks, enable_caching,
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
        self.prefix_cache_manager = RBLNFullAttentionManager(
            kv_cache_spec=self.kv_cache_config.kv_cache_groups[0].
            kv_cache_spec,
            block_pool=self.prefix_caching_block_pool,
            kv_cache_group_id=0,
            caching_hash_fn=caching_hash_fn,
        )
        # [inner_block_id, [outer_block_id, inner_block_offset]]
        self.inner_to_outer_map: dict[int, tuple[int, int]] = {}
        # [outer_block_id, list[inner_block_id]]
        self.outer_to_inner_map: dict[int, list[int]] = {}

    # def find_longest_cache_hit(
    #     self,
    #     block_hashes: list[BlockHash],
    #     max_cache_hit_length: int,
    # ) -> tuple[tuple[list[KVCacheBlock], ...], int]:
    #     """
    #     Get the longest blocks that matched with already cached blocks
    #     and the offset of the last cached block.

    #     Args:
    #         block_hashes: The block hashes of the request.
    #         max_cache_hit_length: The maximum length of tokens
    #             that are available to hit the cache.

    #     Returns:
    #         - A list of cached blocks for each kv cache group
    #         - The number of new tokens that are already computed
    #             (prefix caching)
    #     """
    #     # max_cache_hit_length = request.num_tokens - 1 고정
    #     # NOTE(eunji) change to prefix_cache_manager, prefix_caching_block_pool
    #     inner_hit_blocks = self.prefix_cache_manager.find_longest_cache_hit(
    #         block_hashes=block_hashes,
    #         max_length=max_cache_hit_length,
    #         kv_cache_group_ids=[0],
    #         block_pool=self.prefix_caching_block_pool,
    #         kv_cache_spec=self.kv_cache_spec,
    #         use_eagle=self.use_eagle,
    #     )

    #     # [outer_block_id, inner_block_offset]
    #     dual_block_pool_mapper: dict[int, set[int]] = {}

    #     for inner_hit_block in inner_hit_blocks:
    #         block_outer_idx, block_inner_idx = self.inner_to_outer_map[
    #             inner_hit_block.block_id]
    #         dual_block_pool_mapper[block_outer_idx].add(block_inner_idx)

    #     hit_blocks = list(dual_block_pool_mapper.keys())
    #     num_new_computed_blocks = len(inner_hit_blocks)
    #     return hit_blocks, num_new_computed_blocks * PREFIX_CACHING_BLOCK_SIZE

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
        # NOTE(eunji) inner block 기준으로 리턴
        inner_hit_blocks = self.prefix_cache_manager.find_longest_cache_hit(
            block_hashes=block_hashes,
            max_length=max_cache_hit_length,
            kv_cache_group_ids=[0],
            block_pool=self.prefix_caching_block_pool,
            kv_cache_spec=self.kv_cache_spec,
            use_eagle=self.use_eagle,
        )

        # [outer_block_id, inner_block_offset]
        # dual_block_pool_mapper: dict[int, set[int]] = {}

        # for inner_hit_block in inner_hit_blocks:
        #     block_outer_idx, block_inner_idx = self.inner_to_outer_map[
        #         inner_hit_block.block_id]
        #     dual_block_pool_mapper[block_outer_idx].add(block_inner_idx)

        # hit_blocks = list(dual_block_pool_mapper.keys())
        # num_new_computed_blocks = len(inner_hit_blocks)
        return inner_hit_blocks, inner_hit_blocks * PREFIX_CACHING_BLOCK_SIZE

    def save_new_computed_blocks(
            self, request_id: str,
            new_computed_blocks: tuple[list[KVCacheBlock], ...]) -> None:
        """
        Add the new computed blocks to the request.

        Args:
            request_id: The request ID.
            new_computed_blocks: The new computed blocks just hitting the
                prefix cache.
        """
        for i, manager in enumerate(self.single_type_managers):
            manager.save_new_computed_blocks(request_id,
                                             new_computed_blocks[i])

            # TODO change the outer block id -> inner block id

            self.prefix_cache_manager.save_new_computed_blocks(
                request_id, new_computed_blocks[i])

    def allocate_new_blocks(self, request_id: str,
                            num_tokens: int) -> tuple[list[KVCacheBlock], ...]:
        """
        Allocate new blocks for the request to give it at least `num_tokens` 
        token slots.

        Args:
            request_id: The request ID.
            num_tokens: The total number of tokens that need a slot (including 
                tokens that are already allocated).

        Returns:
            The new allocated blocks.
        """
        outer_new_blocks: tuple[list[KVCacheBlock]] = tuple(
            manager.allocate_new_blocks(request_id, num_tokens)
            for manager in self.single_type_managers)

        inner_new_blocks: tuple[list[KVCacheBlock]] = \
            tuple(
                self.prefix_cache_manager.allocate_new_blocks(
                    request_id, num_tokens
                )
            )

        step = self.block_size // PREFIX_CACHING_BLOCK_SIZE
        for idx in range(0, len(inner_hit_blocks), step):
            start_pos = idx
            end_pos = idx + step
            outer_block_id = idx // PREFIX_CACHING_BLOCK_SIZE
            for inner_block_idx in range(start_pos, end_pos):
                inner_new_block_id = inner_hit_blocks[0][
                    inner_block_idx].block_id
                self.inner_to_outer_map[inner_new_block_id] = tuple(
                    outer_block_id, inner_block_idx - start_pos)

        return outer_new_blocks

    def cache_blocks(self, request: Request, num_computed_tokens: int) -> None:
        """
        Cache the blocks for the request.

        Args:
            request: The request.
            num_computed_tokens: The total number of tokens
                that need to be cached
                (including tokens that are already cached).
        """
        # for manager in self.single_type_managers:
        #     manager.cache_blocks(request, num_computed_tokens)
        self.prefix_cache_manager.cache_blocks(request, num_computed_tokens)

    def free(self, request_id: str) -> None:
        """
        Free the blocks for the request.

        Args:
            request_id: The request ID.
        """
        inner_blocks: tuple[list[KVCacheBlock],
                            ...] = self.get_inner_blocks(request_id)
        for inner_block in inner_blocks[0]:
            inner_block_id = inner_block.block_id
            self.inner_to_outer_map.pop(inner_block_id)

        for manager in self.single_type_managers:
            manager.free(request_id)

        self.prefix_cache_manager.free(request_id)

    def get_num_common_prefix_blocks(self, request_id: str,
                                     num_running_requests: int) -> list[int]:
        """
        Get the number of common prefix blocks for a request.

        Args:
            request_id: The request ID.
            block_hashes: The block hashes of the request.

        Returns:
            The number of common prefix blocks.
        """

        # NOTE(eunji): It is for cascade attention
        # that is not supported in vLLM RBLN
        # num_blocks_per_group = [
        #     manager.get_num_common_prefix_blocks(request_id,
        #                                          num_running_requests)
        #     for manager in self.single_type_managers
        # ]
        # return num_blocks_per_group
        return 0

    def remove_skipped_blocks(self, request_id: str,
                              num_computed_tokens: int) -> None:
        """
        Remove the blocks that are no longer needed from `blocks` and replace 
        the removed blocks with null_block.

        Args:
            request_id: The request ID.
            num_computed_tokens: The number of tokens that have been computed.
        """
        for manager in self.single_type_managers:
            manager.remove_skipped_blocks(request_id, num_computed_tokens)

        self.prefix_cache_manager.remove_skipped_blocks(
            request_id, num_computed_tokens)

    def get_inner_blocks(self,
                         request_id: str) -> tuple[list[KVCacheBlock], ...]:
        """
        Get the blocks for the request.
        """
        return tuple(
            self.prefix_cache_manager.req_to_blocks.get(request_id) or [])

    def convert_outer_to_inner_block(
            self, outer_blocks: list[KVCacheBlock]) -> list[KVCacheBlock]:
        inner_block_ids = []
        for outer_block in outer_blocks:
            outer_block_id = outer_block.block_id
            inner_block_ids.extend(self.inner_to_outer_map[outer_block_id])
        return inner_block_ids


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
