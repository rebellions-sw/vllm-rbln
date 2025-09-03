from collections import deque


class RBLNPrefixBlock:
    block_id: int
    ref_cnt: int = 0


class RBLNPrefixBlockQueue:
    def __init__(self, num_blocks):
        self.num_free_blocks = num_blocks
        self.blocks = deque([RBLNPrefixBlock(block_id=i) for i in range(self.num_free_blocks)])

class PrefixKVCacheManager:
    def __init__(self, vllm_config, num_blocks):
        self.req_to_outer_blocks: dict[str, int]
        self.req_to_inner_blocks: dict[str, int]
        self.inner_to_outer_mapper: dict[int, int]

        self.outer_block_size = vllm_config.additional_config.attn_block_size
        self.inner_block_size = vllm_config.cache_config.block_size

        self.outer_num_blocks = num_blocks
        self.inner_num_blocks \
            = num_blocks * (self.outer_block_size // self.inner_block_size)
        
        self.outer_block_pool = RBLNPrefixBlockQueue(outer_num_blocks)


    # def allocate_outer_block(self, )


    