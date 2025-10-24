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
from collections import deque

import torch


class RBLNPrefixBlockQueue:

    def __init__(self, num_blocks):
        self.num_free_blocks = num_blocks
        self.blocks = deque([i for i in range(self.num_free_blocks)])

    def popleft(self) -> int:
        if len(self.blocks) == 0:
            raise RuntimeError("No free outer blocks available: "
                               "prefix cache block management exhausted.")
        return self.blocks.popleft()

    def append(self, block_id: int) -> None:
        self.blocks.append(block_id)


class RBLNPrefixKVCacheManager:

    def __init__(self, ob_size, ib_size, num_ob):
        self.req_to_outer_blocks: dict[str, list[int]] = {}
        self.pooled_tensor = torch.zeros(1, num_ob, dtype=torch.int32)
        # Check the inner block is oudated or not
        self.inner_to_request_id: dict[int, str] = {}
        self.inner_to_outer_block: dict[int, int] = {}

        self.ob_size = ob_size
        self.ib_size = ib_size
        self.num_ob = num_ob
        self.blk_ratio = self.ob_size // self.ib_size

        self.free_block_queue = RBLNPrefixBlockQueue(self.num_ob)

    def _num_cached_outer_blocks(self, cached_len_tokens: int) -> int:
        return cached_len_tokens // self.ob_size

    def _num_cached_inner_blocks(self, cached_len_tokens: int) -> int:
        return cached_len_tokens // self.ib_size

    def allocate_blocks(self, request_id: str, cached_len: int,
                        inner_blocks: list[int]) -> None:
        """
        Allocate outer blocks for the given inner blocks.
        """
        if request_id not in self.req_to_outer_blocks:
            self.req_to_outer_blocks[request_id] = []

        # Lazy cleanup: finished requests whose inner blocks got reused
        self.free_blocks_of_finished_requests(cached_len, inner_blocks)

        # Allocate the outer blocks that are cached.
        num_cached_outer_blocks = self._num_cached_outer_blocks(cached_len)
        for _ in range(num_cached_outer_blocks):
            new_ob = self.free_block_queue.popleft()
            self.req_to_outer_blocks[request_id].append(new_ob)

        # Allocate the inner blocks that are not cached yet.
        num_cached_ib = self._num_cached_inner_blocks(cached_len)
        uncached_ib = inner_blocks[num_cached_ib:]

        num_new_ob = (len(uncached_ib) + self.blk_ratio - 1) // self.blk_ratio
        ob_idx = 0
        while ob_idx < num_new_ob:
            new_ob = self.free_block_queue.popleft()
            self.req_to_outer_blocks[request_id].append(new_ob)
            start_pos = ob_idx * self.blk_ratio
            end_pos = min((ob_idx + 1) * self.blk_ratio, len(uncached_ib))
            for ib_idx in range(start_pos, end_pos):
                new_ib_id = uncached_ib[ib_idx]
                self.inner_to_outer_block[new_ib_id] = new_ob
                self.inner_to_request_id[new_ib_id] = request_id
            ob_idx += 1

    def free_blocks(self, request_id: str) -> None:
        finished_obs = self.req_to_outer_blocks.pop(request_id)
        for ob in finished_obs:
            self.free_block_queue.append(ob)

    def free_blocks_of_finished_requests(self, cached_len,
                                         inner_blocks: list[int]):
        # 해당 inner block은 그 사이에 free되고 새로 할당되었다
        # 이에 맞추어 outer block도 해제
        """
        Free the outer blocks of finished requests.
        """
        num_cached_ib = self._num_cached_inner_blocks(cached_len)
        uncached_ib = inner_blocks[num_cached_ib:]

        for ib_id in uncached_ib:
            if ib_id in self.inner_to_request_id:
                request_id = self.inner_to_request_id.pop(ib_id)
                if request_id in self.req_to_outer_blocks:
                    self.free_blocks(request_id)

    def get_cached_origin_blocks(self, cached_len,
                                 inner_blocks: list[int]) -> torch.Tensor:
        """
        Get the outer blocks that are already cached.
        """
        cached_outer_blocks = []
        last_cached_outer_block = None

        num_cached_ib = self._num_cached_inner_blocks(cached_len)
        cached_ib = inner_blocks[:num_cached_ib]

        for ib_id in cached_ib:
            ob_id = self.inner_to_outer_block[ib_id]
            if ob_id != last_cached_outer_block:
                cached_outer_blocks.append(ob_id)   
                last_cached_outer_block = ob_id
        if cached_outer_blocks:
            return torch.tensor(cached_outer_blocks, dtype=torch.int32)
        else:
            return None

    def get_blocks(self, request_id: str) -> torch.Tensor:
        """
        Get all the outer blocks allocated to the given request.
        """
        self.pooled_tensor.fill_(-1)
        value =  torch.tensor(self.req_to_outer_blocks[request_id], dtype=torch.int16)
        self.pooled_tensor[0, :len(value)].copy_(value)
        return self.pooled_tensor
