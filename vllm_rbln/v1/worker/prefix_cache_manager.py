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


class RBLNBlock:

    def __init__(self, block_id: int):
        self.block_id = block_id


class RBLNBlockManager:

    def __init__(self, num_blocks):
        self.num_blocks = num_blocks
        self.blocks = deque(
            [RBLNBlock(block_id=i) for i in range(self.num_blocks)])

    def popleft(self) -> RBLNBlock:
        if len(self.blocks) == 0:
            raise RuntimeError("No free outer blocks available: "
                               "prefix cache block management exhausted.")

        block = self.blocks.popleft()
        return block

    def append(self, block: RBLNBlock) -> None:
        self.blocks.append(block)


class RBLNPrefixKVCacheManager:
    """
    Compared to the original prefix caching in vLLM, this implementation:
    1. Operates on outer blocks rather than inner blocks.
    2. Copies cached blocks into new blocks instead of reusing them directly.
    3. Frees Asynchronously
    - outer_to_inner_blocks, inner_to_outer_block
        - Frees outer blocks when the inner blocks mapped to them are newly allocated to other requests.
    - req_to_outer_blocks
        - Frees outer blocks when the request is finished.
    """

    def __init__(self, ob_size: int, ib_size: int, max_model_len: int,
                 num_ob: int):
        self.req_to_outer_blocks: dict[str, list[int]] = {}
        self.inner_to_outer_block: dict[int, int] = {}
        self.outer_to_inner_blocks: [tuple[RBLNBlock, list[int]]
                                     ] = [tuple() for _ in range(num_ob)]

        self.pooled_tensor = torch.zeros(1,
                                         max_model_len // ob_size,
                                         dtype=torch.int32)

        self.ob_size = ob_size
        self.ib_size = ib_size
        self.num_ob = num_ob
        assert ob_size % ib_size == 0, "ob_size must be a multiple of ib_size"
        self.blk_ratio = self.ob_size // self.ib_size

        self.outer_block_manager = RBLNBlockManager(self.num_ob)

    def _num_cached_outer_blocks(self, cached_len_tokens: int) -> int:
        return cached_len_tokens // self.ob_size

    def _num_cached_inner_blocks(self, cached_len_tokens: int) -> int:
        return cached_len_tokens // self.ib_size

    def _allocate_new_ob(self) -> int:
        new_ob = self.outer_block_manager.popleft()
        return new_ob

    def _allocate_ibs_per_ob(self, new_ob: RBLNBlock, ob_idx: int,
                             uncached_ib: list[int]) -> None:
        start_pos = ob_idx * self.blk_ratio
        end_pos = min((ob_idx + 1) * self.blk_ratio, len(uncached_ib))
        for ib_idx in range(start_pos, end_pos):
            new_ib_id = uncached_ib[ib_idx]
            self.inner_to_outer_block[new_ib_id] = new_ob.block_id

        self.outer_to_inner_blocks[new_ob.block_id] = (
            new_ob, uncached_ib[start_pos:end_pos])

    def allocate_blocks(self, request_id: str, cached_len: int,
                        inner_blocks: list[int]) -> None:
        """
        Allocate outer blocks for the given inner blocks.
        """
        num_cached_ib = self._num_cached_inner_blocks(cached_len)
        uncached_ib = inner_blocks[num_cached_ib:]
        # Lazy free of the outer blocks whose inner blocks are newly allocated.
        self._free_blocks(uncached_ib)

        if request_id not in self.req_to_outer_blocks:
            self.req_to_outer_blocks[request_id] = []

        # Allocate the outer blocks that are cached.
        num_cached_outer_blocks = self._num_cached_outer_blocks(cached_len)
        for _ in range(num_cached_outer_blocks):
            new_ob = self._allocate_new_ob()
            self.req_to_outer_blocks[request_id].append(new_ob.block_id)

        # Allocate the outer blocks that are not cached yet.
        # Map the inner blocks to the new outer blocks.
        num_new_ob = (len(uncached_ib) + self.blk_ratio - 1) // self.blk_ratio
        ob_idx = 0
        while ob_idx < num_new_ob:
            new_ob = self._allocate_new_ob()
            self.req_to_outer_blocks[request_id].append(new_ob.block_id)
            self._allocate_ibs_per_ob(new_ob, ob_idx, uncached_ib)
            ob_idx += 1

    def _free_blocks(self, inner_blocks: list[int]) -> None:
        """
        Free the outer blocks whose inner blocks
        are newly allocated to other requests.
        """
        for ib in inner_blocks:
            ob_id = self.inner_to_outer_block.get(ib, None)
            if ob_id is None:
                # It is already freed.
                continue
            ob, ibs = self.outer_to_inner_blocks[ob_id]
            self.outer_block_manager.append(ob)
            for ib in ibs:
                self.inner_to_outer_block.pop(ib)
            self.outer_to_inner_blocks[ob_id] = tuple()

    def free_request(self, request_id: str) -> None:
        self.req_to_outer_blocks.pop(request_id, None)

    def get_cached_origin_blocks(self, request_id, cached_len,
                                 inner_blocks: list[int]) -> torch.Tensor:
        """
        Get the outer blocks that are already cached.
        """
        cached_outer_blocks = []
        last_cached_outer_block = None

        num_cached_ib = self._num_cached_inner_blocks(cached_len)
        cached_ib = inner_blocks[:num_cached_ib]

        for ib_id in cached_ib:
            ob_id = self.inner_to_outer_block.get(ib_id)
            if ob_id is None:
                raise RuntimeError(
                    f"Inner block {ib_id} is not mapped to any outer block.")
            if ob_id != last_cached_outer_block:
                cached_outer_blocks.append(ob_id)
                last_cached_outer_block = ob_id

        if cached_outer_blocks:
            return torch.tensor(cached_outer_blocks, dtype=torch.int32)
        return None

    def get_blocks(self, request_id: str) -> torch.Tensor:
        """
        Get all the outer blocks allocated to the given request.
        """
        self.pooled_tensor.fill_(-1)
        value = torch.tensor(self.req_to_outer_blocks[request_id],
                             dtype=torch.int32)
        self.pooled_tensor[0, :len(value)].copy_(value)
        return self.pooled_tensor
