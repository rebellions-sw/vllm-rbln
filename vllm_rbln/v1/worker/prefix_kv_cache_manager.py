from collections import deque


class RBLNPrefixBlock:
    block_id: int
    ref_cnt: int = 0

    def incr_ref(self):
        self.ref_cnt += 1
    
    def decr_ref(self):
        self.ref_cnt -= 1

class RBLNPrefixBlockQueue:
    def __init__(self, num_blocks):
        self.num_free_blocks = num_blocks
        self.blocks = deque([RBLNPrefixBlock(block_id=i) for i in range(self.num_free_blocks)])

    def popleft(self) -> int:
        return self.blocks.popleft()

    def append(self, block) -> None:
        self.blocks.append(block)
        

class PrefixKVCacheManager:
    def __init__(self, vllm_config, num_blocks):
        self.req_to_outer_blocks: dict[str, list[int]]
        # self.req_to_inner_blocks: dict[str, int]
        self.inner_to_outer_mapper: dict[int, int]
        self.outer_to_inner_mapper: dict[int, list[int]]

        self.outer_blk_size = vllm_config.additional_config.attn_block_size
        self.inner_blk_size = vllm_config.cache_config.block_size
        self.blk_ratio = self.outer_blk_size // self.inner_blk_size

        self.outer_num_blks = num_blocks
        self.inner_num_blks = num_blocks * self.blk_ratio
        
        self.free_block_queue = RBLNPrefixBlockQueue(self.outer_num_blks)


    def allocate_outer_block(self, request_id, new_inner_blocks: list[int]) -> list[int]:
        """
        Allocate outer_blocks following new_inner_blocks
        """
        num_new_outer_blocks = (len(new_inner_blocks) + self.blk_ratio - 1) // self.blk_ratio
        ret: list[int] = []
        idx = 0
        while idx < num_new_outer_blocks:
            # inner_block = new_inner_blocks
            curr_block = self.free_block_queue.popleft()
            start_pos = idx * self.blk_ratio
            end_pos = (idx + 1) * self.blk_ratio
            for inner_idx in range(start_pos, end_pos):
                inner_blk_id =  new_inner_blocks[inner_idx]
                self.inner_to_outer_mapper[inner_blk_id] = curr_block
                self.outer_to_inner_mapper[curr_block.block_id].append(inner_blk_id)
            ret.append(curr_block.block_id)
            self.req_to_outer_blocks[request_id].append(curr_block.block_id)
            idx += 1
        return ret

    def free_outer_block(self, request_id, inner_blocks: list[int]) -> None:
        inner_blk_id = inner_blocks[0]
        outer_blk_id = self.inner_to_outer_mapper[inner_blk_id]
        self.outer_to_inner_mapper.pop(outer_blk_id)
        self.req_to_outer_blocks.pop(request_id)

        for inner_blk_id in inner_blocks:
            curr_block = self.inner_to_outer_mapper.pop(inner_blk_id)
            curr_block.decr_ref()
            if curr_block.ref_cnt == 0:
                self.free_block_queue.append(curr_block)

    def get_outer_blocks(self, request_id) -> list[int]:
        outer_blk_ids = self.req_to_outer_blocks[request_id]
        return outer_blk_ids

        


    