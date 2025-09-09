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
        self.ref_cnt_per_outer_block = [0] * num_blocks

    def popleft(self) -> int:
        if len(self.blocks) == 0:
            raise RuntimeError("No free outer blocks available: "
                               "prefix cache block management exhausted.")
        block_id = self.blocks.popleft()
        self.ref_cnt_per_outer_block[block_id] = 1  # set ref count to 1
        return block_id

    def append(self, block_id: int) -> None:
        self.ref_cnt_per_outer_block[block_id] += 1  # increase ref count
        self.blocks.append(block_id)

    def get_ref_cnt(self, block_id: int) -> int:
        return self.ref_cnt_per_outer_block[block_id]

    def reset_ref_cnt(self, block_id: int) -> None:
        self.ref_cnt_per_outer_block[block_id] = 0

    def inc_ref_cnt(self, block_id: int) -> None:
        self.ref_cnt_per_outer_block[block_id] += 1


class RBLNPrefixKVCacheManager:

    def __init__(self, ob_size: int, ib_size: int, max_model_len: int,
                 num_ob: int):
        self.req_to_outer_blocks: dict[str, list[int]] = {}
        self.pooled_tensor = torch.zeros(1,
                                         max_model_len // ob_size,
                                         dtype=torch.int32)
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
        # self.free_blocks_of_finished_requests(cached_len, inner_blocks)

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
        """
        Free the outer blocks allocated to the given request.
        It only frees the outer blocks whose reference count is 0.
        """
        obs = self.req_to_outer_blocks.get(request_id)
        num_freed_obs = 0
        # To sync with the order of block pool
        for ob in reversed(obs):
            if self.get_ref_cnt(ob) == 1:
                self.reset_ref_cnt(ob)
                self.free_block_queue.append(ob)
                num_freed_obs += 1
        if num_freed_obs == len(obs):
            del self.req_to_outer_blocks[request_id]
        else:
            # .remove() is expensive, so we create a new list
            self.req_to_outer_blocks[request_id] = [
                ob for ob in obs if self.get_ref_cnt(ob) > 0
            ]

    def get_cached_origin_blocks(self, cached_len,
                                 inner_blocks: torch.Tensor) -> torch.Tensor:
        """
        Get the outer blocks that are already cached.
        It must be called after allocate_blocks()
        that calls free_blocks_of_finished_requests().
        """
        cached_outer_blocks = []
        last_cached_outer_block = None

        num_cached_ib = self._num_cached_inner_blocks(cached_len)
        cached_ib = inner_blocks[:num_cached_ib]

        for ib_id in cached_ib:
            ob_id = self.inner_to_outer_block[ib_id]
            if ob_id != last_cached_outer_block:
                self.inc_ref_cnt(ob_id)
                print("inc ref cnt of block", ob_id, "to",
                      self.get_ref_cnt(ob_id))
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
        value = torch.tensor(self.req_to_outer_blocks[request_id],
                             dtype=torch.int16)
        self.pooled_tensor[0, :len(value)].copy_(value)
        return self.pooled_tensor

    def get_ref_cnt(self, block_id: int) -> int:
        return self.free_block_queue.get_ref_cnt(block_id)

    def reset_ref_cnt(self, block_id: int) -> None:
        self.free_block_queue.reset_ref_cnt(block_id)

    def inc_ref_cnt(self, block_id: int) -> None:
        self.free_block_queue.inc_ref_cnt(block_id)
