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
from vllm.v1.core.block_pool import BlockPool

class RBLNBlockPool(BlockPool):


    def __init__(
        self,
        num_gpu_blocks: int,
        enable_caching: bool,
        enable_kv_cache_events: bool = False,
    ):
        assert isinstance(num_gpu_blocks, int) and num_gpu_blocks > 0
        self.num_gpu_blocks = num_gpu_blocks
        self.enable_caching = enable_caching
        # All kv-cache blocks.
        self.blocks: list[KVCacheBlock] = [
            KVCacheBlock(idx) for idx in range(num_gpu_blocks)
        ]
        # Free block queue that constructs and manipulates a doubly linked
        # list of free blocks (including eviction candidates when caching is
        # enabled).
        self.free_block_queue = FreeKVCacheBlockQueue(self.blocks)

        # {block_hash: {block ID: block}}. A cached block is
        # a full block with a block hash that can be used for prefix caching.
        # The cached block may be used by running requests or in the
        # free_block_queue that could potentially be evicted.
        # NOTE: We currently don't de-duplicate the blocks in the cache,
        # meaning that if a block becomes full and is cached, we don't check
        # if there is already an identical block in the cache. This is because
        # we want to make sure the allocated block IDs won't change so that
        # block tables are append-only.
        self.cached_block_hash_to_block: dict[RBLNBlockHashWithGroupId, dict[
            int, KVCacheBlock]] = defaultdict(dict)

        # To represent a placeholder block with block_id=0.
        # The ref_cnt of null_block is not maintained, needs special care to
        # avoid freeing it.
        self.null_block = self.free_block_queue.popleft()
        self.null_block.is_null = True

        self.enable_kv_cache_events = enable_kv_cache_events
        self.kv_event_queue: list[KVCacheEvent] = []


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
            cached_blocks_one_group = self.cached_block_hash_to_block.get(
                RBLNBlockHashWithGroupId(block_hash, group_id))
            if not cached_blocks_one_group:
                return None
            first_block = next(iter(cached_blocks_one_group.values()))
            cached_blocks.append(first_block)
        return cached_blocks

    def cache_full_blocks(
        self,
        request: Request,
        blocks: list[KVCacheBlock],
        block_hashes: list[BlockHash],
        num_cached_blocks: int,
        num_full_blocks: int,
        block_size: int,
        kv_cache_group_id: int,
        hash_fn: Callable,
    ) -> None:
        """Cache a list of full blocks for prefix caching.
        This function takes a list of blocks that will have their block hash
        metadata to be updated and cached. Given a request, it computes the
        block hashes for the blocks starting from `num_cached_blocks` to
        `num_full_blocks`, updating the metadata for each block
        and caching them in the `cached_block_hash_to_block`.

        Args:
            request: The request to cache the blocks.
            blocks: All blocks in the request.
            block_hashes: Block hashes of the blocks in the request. Note that
            this list may be shorter than the blocks list. In this case the
            missed block hash will be computed in this function.
            num_cached_blocks: The number of blocks that are already cached.
            num_full_blocks: The number of blocks that are full and should
                be cached after this function.
            block_size: Number of tokens in each block.
            kv_cache_group_id: The id of the KV cache group.
            hash_fn: The hash function to use for block hashes.
        """
        if num_cached_blocks == num_full_blocks:
            return
        new_full_blocks = blocks[num_cached_blocks:num_full_blocks]
        assert len(block_hashes) >= num_cached_blocks
        new_block_hashes = block_hashes[num_cached_blocks:]

        # Update the new blocks with the block hashes through the chain.
        if num_cached_blocks == 0:
            prev_block_hash_value = None
        else:
            prev_block = blocks[num_cached_blocks - 1]
            assert prev_block.block_hash is not None
            prev_block_hash_value = prev_block.block_hash.get_hash_value()

        parent_block_hash = prev_block_hash_value
        new_hashes: Optional[list[int]] = ([] if self.enable_kv_cache_events
                                           else None)
        # FIXME indexing become changed
        for i, blk in enumerate(new_full_blocks):
            assert blk.block_hash is None

            j = i * (block_size // PREFIX_CACHING_BLOCK_SIZE)
            if i < len(new_block_hashes):
                # The block hash may already be computed in
                # "get_computed_blocks" if the tokens are not generated by
                # this request (either the prompt tokens or the previously
                # generated tokens with preemption), or by other
                # single_type_managers with the same block_size.
                # In this case we simply reuse the block hash.
                block_hash = new_block_hashes[i]
            else:
                # Otherwise compute the block hash and cache it in the request
                # in case it will be preempted in the future.
                blk_idx = num_cached_blocks + i
                start_token_idx = blk_idx * block_size
                end_token_idx = (blk_idx + 1) * block_size
                block_tokens = request.all_token_ids[
                    start_token_idx:end_token_idx]
                assert len(block_tokens) == block_size, (
                    f"Expected {block_size} tokens, got "
                    f"{len(block_tokens)} at {blk_idx}th block for request "
                    f"{request.request_id}({request})")

                # Generate extra keys for multi-modal inputs. Note that since
                # we reach to this branch only when the block is completed with
                # generated tokens, we only need to consider the last mm input.
                extra_keys, _ = generate_block_hash_extra_keys(
                    request, start_token_idx, end_token_idx, -1)

                # Compute the hash of the current block.
                block_hash = hash_block_tokens(hash_fn, prev_block_hash_value,
                                               block_tokens, extra_keys)
                block_hashes.append(block_hash)

            # Update and added the full block to the cache.
            block_hash_with_group_id = BlockHashWithGroupId(
                block_hash, kv_cache_group_id)
            blk.block_hash = block_hash_with_group_id
            self.cached_block_hash_to_block[block_hash_with_group_id][
                blk.block_id] = blk
            if new_hashes is not None:
                new_hashes.append(block_hash.hash_value)
            prev_block_hash_value = block_hash.hash_value

        if self.enable_kv_cache_events:
            self.kv_event_queue.append(
                BlockStored(
                    block_hashes=new_hashes,
                    parent_block_hash=parent_block_hash,
                    token_ids=request.
                    all_token_ids[num_cached_blocks *
                                  block_size:num_full_blocks * block_size],
                    block_size=block_size,
                    lora_id=request.lora_request.id
                    if request.lora_request else None,
                ))