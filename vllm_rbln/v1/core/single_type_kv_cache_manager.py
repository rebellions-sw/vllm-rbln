from vllm_rbln.v1.kv_cache_interface import RBLNSlidingWindowImageSpec, RBLNSlidingWindowSpec
from vllm.v1.kv_cache_interface import FullAttentionSpec, KVCacheSpec
from vllm.v1.core.single_type_kv_cache_manager import SingleTypeKVCacheManager, FullAttentionManager
from vllm.v1.core.block_pool import BlockPool
from vllm.v1.core.kv_cache_utils import BlockHash, KVCacheBlock
from vllm.v1.request import Request
from vllm.utils import cdiv
from vllm_rbln.logger import init_logger

logger = init_logger(__name__)

class RBLNSlidingWindowImageManager(SingleTypeKVCacheManager):

    def __init__(self, kv_cache_spec: RBLNSlidingWindowImageSpec, block_pool: BlockPool,
                 **kwargs) -> None:
        super().__init__(kv_cache_spec, block_pool, **kwargs)
        self.sliding_window = kv_cache_spec.sliding_window
        self._null_block = block_pool.null_block

    def get_num_blocks_to_allocate(
            self, request_id: str, num_tokens: int,
            new_computed_blocks: list[KVCacheBlock]) -> int:
        # NOTE for local table id
        return super().get_num_blocks_to_allocate(request_id, num_tokens, new_computed_blocks) + 1

    def allocate_new_blocks(self, request_id: str,
                            num_tokens: int) -> list[KVCacheBlock]:
        req_blocks = self.req_to_blocks[request_id]
        # NOTE for local table id
        num_required_blocks = cdiv(num_tokens, self.block_size) + 1
        num_new_blocks = num_required_blocks - len(req_blocks)
        if num_new_blocks <= 0:
            return []
        else:
            new_blocks = self.block_pool.get_new_blocks(num_new_blocks)
            req_blocks.extend(new_blocks)
            return new_blocks

    def cache_blocks(self, request: Request, block_hashes: list[BlockHash],
                     num_tokens: int) -> None:
        """
        Cache the blocks for the request.

        Args:
            request: The request.
            block_hashes: The block hashes of the request.
            num_tokens: The total number of tokens that need to be cached 
                (including tokens that are already cached).
        """
        num_cached_blocks = self.num_cached_block[request.request_id]
        num_full_blocks = num_tokens // self.block_size + 1

        self.block_pool.cache_full_blocks(
            request=request,
            blocks=self.req_to_blocks[request.request_id],
            block_hashes=block_hashes,
            num_cached_blocks=num_cached_blocks,
            num_full_blocks=num_full_blocks,
            block_size=self.block_size,
            kv_cache_group_id=self.kv_cache_group_id,
            hash_fn=self.caching_hash_fn,
        )

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
        """
        Copy of find_longest_cache_hit in FullAttentionManager
        """
        raise RuntimeError("Prefix caching is not supported in vLLM RBLN yet.")
        computed_blocks: tuple[list[KVCacheBlock], ...] = tuple(
            [] for _ in range(len(kv_cache_group_ids)))
        max_num_blocks = max_length // kv_cache_spec.block_size
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

    def remove_skipped_blocks(self, request_id: str,
                              num_computed_tokens: int) -> None:
        # No need to remove blocks for full attention.
        pass

    def get_num_common_prefix_blocks(self, request_id: str,
                                     num_running_requests: int) -> int:
        return 0


spec_manager_map: dict[type[KVCacheSpec], type[SingleTypeKVCacheManager]] = {
    FullAttentionSpec: FullAttentionManager,
    RBLNSlidingWindowSpec: FullAttentionManager,
    RBLNSlidingWindowImageSpec: RBLNSlidingWindowImageManager,
}

def get_manager_for_kv_cache_spec(kv_cache_spec: KVCacheSpec,
                                  **kwargs) -> SingleTypeKVCacheManager:
    manager_class = spec_manager_map[type(kv_cache_spec)]
    manager = manager_class(kv_cache_spec, **kwargs)
    return manager