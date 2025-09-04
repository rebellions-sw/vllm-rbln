from collections import deque
import torch

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
        self.inner_to_outer_blocks: dict[int, int]

        self.outer_blk_size = vllm_config.additional_config.attn_block_size
        self.inner_blk_size = vllm_config.cache_config.block_size
        self.blk_ratio = self.outer_blk_size // self.inner_blk_size

        self.outer_num_blks = num_blocks
        self.inner_num_blks = num_blocks * self.blk_ratio
        
        self.free_block_queue = RBLNPrefixBlockQueue(self.outer_num_blks)


    def allocate_blocks(self, request_id:str, new_inner_blocks: list[int]) -> None:
        """
        Allocate outer_blocks following new_inner_blocks
        """
        num_new_outer_blocks = (len(new_inner_blocks) + self.blk_ratio - 1) // self.blk_ratio
        idx = 0
        while idx < num_new_outer_blocks:
            # inner_block = new_inner_blocks
            curr_block = self.free_block_queue.popleft()
            self.req_to_outer_blocks[request_id].append(curr_block.block_id)
            start_pos = idx * self.blk_ratio
            end_pos = (idx + 1) * self.blk_ratio
            for idx2 in range(start_pos, end_pos):
                curr_inner_block = new_inner_blocks[idx2]
                self.inner_to_outer_blocks[curr_inner_block] = curr_block
            idx += 1

    def allocate_prefill_blocks(self, request_id:str, new_inner_blocks: list[int]) -> None:
        """
        Allocate outer_blocks following new_inner_blocks
        """
        cached_outer_blocks = []
        num_cached_inner_blocks = 0
        # 1. Find prefix-cached blocks
        for new_inner_block in new_inner_blocks:
            if new_inner_block in self.inner_to_outer_blocks.keys():
                num_cached_inner_blocks += 1
                cached_outer_block = self.inner_to_outer_blocks[new_inner_block]
                if cached_outer_block == cached_outer_blocks[-1]:
                    pass
                else:
                    cached_outer_blocks.append(cached_outer_block)
                    new_block = self.free_block_queue.popleft()
                    self.req_to_outer_blocks[request_id].append(new_block)
            else:
                break
        new_inner_blocks = new_inner_blocks[num_cached_inner_blocks:]

        # 2. Allocate new blocks that are not prefix cached.
        num_new_outer_blocks = (len(new_inner_blocks) + self.blk_ratio - 1) // self.blk_ratio
        idx = 0
        while idx < num_new_outer_blocks:
            # inner_block = new_inner_blocks
            curr_block = self.free_block_queue.popleft()
            self.req_to_outer_blocks[request_id].append(curr_block.block_id)
            start_pos = idx * self.blk_ratio
            end_pos = (idx + 1) * self.blk_ratio
            for idx2 in range(start_pos, end_pos):
                curr_inner_block = new_inner_blocks[idx2]
                self.inner_to_outer_blocks[curr_inner_block] = curr_block
            idx += 1
        return cached_outer_blocks, new_outer_blocks

    def get_blocks(self, request_id: str) -> torch.Tensor:
        return torch.tensor(self.req_to_outer_blocks[request_id])

    def free_blocks(self, request_id: str) -> None:
        self.req_to_outer_blocks.pop(request_id)
