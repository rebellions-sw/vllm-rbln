from collections import deque
import torch

class RBLNPrefixBlockQueue:
    def __init__(self, num_blocks):
        self.num_free_blocks = num_blocks
        self.blocks = deque([i for i in range(self.num_free_blocks)])

    def popleft(self) -> int:
        return self.blocks.popleft()

    def append(self, block_id: int) -> None:
        self.blocks.append(block_id)

class PrefixKVCacheManager:
    def __init__(self, vllm_config, num_blocks):
        self.req_to_outer_blocks: dict[str, list[int]]
        self.inner_to_request_id: dict[int, str] # inner block의 outdated 체크 목적 
        self.inner_to_outer_block: dict[int, int]

        self.outer_blk_size = vllm_config.additional_config.attn_block_size
        self.inner_blk_size = vllm_config.cache_config.block_size
        self.blk_ratio = self.outer_blk_size // self.inner_blk_size

        self.outer_num_blks = num_blocks
        self.inner_num_blks = num_blocks * self.blk_ratio
        
        self.free_block_queue = RBLNPrefixBlockQueue(self.outer_num_blks)
        
    def allocate_blocks(self, request_id:str, cached_len: int, new_inner_blocks: list[int] = []) -> None:
        """
        Allocate outer_blocks following new_inner_blocks
        """
        if request_id not in self.req_to_outer_blocks:
            self.req_to_outer_blocks[request_id] = []
        
        # Lazy update - Remove the already finished requests
        # mark the inner block of the requests
        free_blocks_of_finished_requests(cached_len, new_inner_blocks)
        
        # Allocate the outer blocks that are cached.
        num_cached_outer_blocks = cached_len // self.outer_blk_size 
        for _ in num_cached_outer_blocks:
            new_ob = self.free_block_queue.popleft()
            self.req_to_outer_blocks[request_id].append(new_ob)

        # Allocate the inner blocks that are not cached yet.
        num_cached_inner_blocks = cached_len * self.inner_blk_size
        new_inner_blocks = new_inner_blocks[num_cached_inner_blocks:]
        
        num_new_outer_blocks = (len(new_inner_blocks) + self.blk_ratio - 1) // self.blk_ratio
        ob_idx = 0
        while ob_idx < num_new_outer_blocks:
            new_ob = self.free_block_queue.popleft()
            self.req_to_outer_blocks[request_id].append(new_ob)
            start_pos = ob_idx * self.blk_ratio
            end_pos = min((ob_idx + 1) * self.blk_ratio, len(new_inner_blocks))
            for ib_idx in range(start_pos, end_pos):
                new_ib_id = new_inner_blocks[ib_idx]
                self.inner_to_outer_block[new_ib_id] = new_ob.block_id
            ob_idx += 1

    def free_blocks(self, request_id: str) -> None:
        outer_blocks = self.req_to_outer_blocks.pop(request_id)
        for ob in outer_blocks:
            self.free_block_queue.append(ob)

    def free_blocks_of_finished_requests(self, cached_len, inner_blocks: list[int]):
        # 해당 inner block은 그 사이에 free되고 새로 할당되었다
        # 이에 맞추어 outer block도 해제
        num_cached_ib = cached_len // self.inner_blk_size
        target_ibs = inner_blocks[num_cached_ib:]
        for ib_idx in target_ibs:
            ib_id = inner_blocks[ib_idx]
            if ib_id in self.inner_to_request_id:
                request_id = self.inner_to_request_id.pop(ib_id)
                if request_id in self.req_to_outer_blocks:
                    self.free_blocks(request_id)

    def get_cached_origin_blocks(self, cached_len, inner_blocks: list[int]) -> torch.Tensor:
        cached_outer_blocks = []
        last_cached_outer_block = None

        num_cached_ib = cached_len // self.inner_blk_size
        target_ibs = inner_blocks[:num_cached_ib]

        for ib_idx in target_ibs:
            ib_id = inner_blocks[ib_idx]
            ob_id = self.inner_to_outer_block[ib_id]
            cached_outer_blocks.append(ob_id)
            last_cached_outer_block = ob_id
        return torch.tensor(cached_outer_blocks)

    def get_blocks(self, request_id: str) -> torch.Tensor:
        return torch.tensor(self.req_to_outer_blocks[request_id])