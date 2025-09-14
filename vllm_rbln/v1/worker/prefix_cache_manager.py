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

from vllm_rbln.logger import init_logger

logger = init_logger(__name__)


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
    
    def is_empty(self) -> bool:
        return len(self.blocks) == 0


class RBLNPrefixKVCacheManager:
    """
    Compared to the original prefix caching in vLLM, this implementation:
    1. Operates on outer blocks rather than inner blocks.
    2. Copies cached blocks into new blocks instead of reusing them directly.
    3. Frees Asynchronously
    - outer_to_inner_blocks, inner_to_outer_block
        - Frees outer blocks when the inner blocks
        mapped to them are newly allocated to other requests.
    - req_to_outer_blocks, outer_block_to_req
        - Frees outer blocks when the request is finished.

    Note
    - Eviction policy:
        - If no free outer blocks, evict outer blocks
            whose finished requests in round-robin manner.
    - Life cycle:
        1. Request != Inner blocks in vLLM
            - Even though a request is finished,
                its inner blocks may be reused by other requests.
            - So we keep the inner blocks alive
                until they are reused by other requests.
        2. Inner blocks in vLLM <-> Outer blocks in RBLN
            - Inner blocks are cached in vLLM, but outer blocks
                can be evicted in RBLN if free outer blocks are needed.
            - So we evict outer blocks whose inner blocks are cached in vLLM.
    """

    def __init__(self, ob_size: int, ib_size: int, max_model_len: int,
                 num_ob: int):
        self.req_to_outer_blocks: dict[str, list[int]] = {}
        self.outer_block_to_req: dict[int, str] = {}
        # TODO 1 inner block to multiple outer blocks?
        self.inner_to_outer_block: dict[int, int] = {}
        # TODO contained cached inner blocks?
        self.outer_to_inner_blocks: [tuple[RBLNBlock, list[int]]
                                     ] = [tuple() for _ in range(num_ob)]

        self.pooled_tensor = torch.zeros(max_model_len // ob_size,
                                         dtype=torch.int32)

        self.ob_size = ob_size
        self.ib_size = ib_size
        self.num_ob = num_ob
        assert ob_size % ib_size == 0, "ob_size must be a multiple of ib_size"
        self.blk_ratio = self.ob_size // self.ib_size

        self.outer_block_manager = RBLNBlockManager(self.num_ob)

    def _num_cached_inner_blocks(self, cached_len_tokens: int) -> int:
        return cached_len_tokens // self.ib_size

    def _evict_uncached_ib(self) -> None:
        """
        Evict the outer blocks of finished requests
        if no free outer blocks are available.
        TODO optimize the eviction policy.
        TODO sub prefix caching
        """
        for ob_id, (ob, ibs) in enumerate(self.outer_to_inner_blocks):
            if len(ibs) == 0:
                continue
            if ob_id not in self.outer_block_to_req:
                self.outer_block_manager.append(ob)
                logger.debug("[PFX] [EVICTION] OB=%d (IB=%s)", ob_id, ibs)
                for ib in ibs:
                    self.inner_to_outer_block.pop(ib, None)
                self.outer_to_inner_blocks[ob_id] = tuple()

    def _allocate_new_ob(self) -> int:
        """
        Allocate a new outer block.
        If no free outer blocks, evict the outer blocks of finished requests.
        """
        if self.outer_block_manager.is_empty():
            self._evict_uncached_ib()
        new_ob = self.outer_block_manager.popleft()
        return new_ob

    def _allocate_ibs_per_ob(self, new_ob: RBLNBlock, ob_idx: int,
                             uncached_ib: list[int]) -> None:
        for ib_id in uncached_ib:
            self.inner_to_outer_block[ib_id] = new_ob.block_id

        self.outer_to_inner_blocks[new_ob.block_id] = (
            new_ob, uncached_ib)

    def _append_new_ib(self, last_ob_id: int, inner_blocks: list[int]) -> None:
        """
        Append new inner blocks to the last outer block of the request.
        """
        assert len(inner_blocks) == 1
        new_ib = inner_blocks[0]
        self.inner_to_outer_block[new_ib] = last_ob_id
        self.outer_to_inner_blocks[last_ob_id][1].append(new_ib)

    def allocate_blocks(self, request_id: str, cached_len: int,
                        inner_blocks: list[int]) -> None:
        """
        Allocate outer blocks for the given inner blocks.
        """
        print("request_id:", request_id)
        print("cached_len:", cached_len)
        print("inner_blocks:", inner_blocks)
        if request_id in self.req_to_outer_blocks:
            num_already_allocated_ibs = cached_len // self.ib_size
            if num_already_allocated_ibs % self.blk_ratio == 0:
                num_obs = 1
            else:
                last_ob_id = self.req_to_outer_blocks[request_id][-1]
                self._append_new_ib(last_ob_id, inner_blocks)
                return
        else:
            self.req_to_outer_blocks[request_id] = []
            num_obs = (len(inner_blocks) + self.blk_ratio - 1) // self.blk_ratio
        
        for ob_idx in range(num_obs):
            new_ob = self._allocate_new_ob()
            # Map the new outer block to the request.
            self.req_to_outer_blocks[request_id].append(new_ob.block_id)
            self.outer_block_to_req[new_ob.block_id] = request_id

            # Map the new outer block to the inner blocks.
            start_pos = ob_idx * self.blk_ratio
            end_pos = min((ob_idx + 1) * self.blk_ratio, len(inner_blocks))
            new_ibs = []
            for ib_id in inner_blocks[start_pos:end_pos]:
                if ib_id not in self.inner_to_outer_block:
                    new_ibs.append(ib_id)
            self._allocate_ibs_per_ob(new_ob, ob_idx, new_ibs)

        obs = self.req_to_outer_blocks[request_id]
        logger.debug("[PFX] [ALLOC] REQUEST=%s OB=%s (IB=%s)", request_id, obs, inner_blocks)

    def free_request(self, request_id: str) -> None:
        """
        Remove the mapping of the given request.
        Keep the outer blocks alive until their inner blocks
        are reused by other requests.
        """
        outer_blocks = self.req_to_outer_blocks.pop(request_id, None)
        for ob_id in outer_blocks:
            self.outer_block_to_req.pop(ob_id, None)

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
            assert ob_id is not None, (
                "Inconsistent state: cached inner block"
            )
            # It is cached in vLLM, but not in RBLN.
            req = self.outer_block_to_req.get(ob_id)
            if req == request_id:
                break
            if ob_id != last_cached_outer_block:
                cached_outer_blocks.append(ob_id)
                last_cached_outer_block = ob_id

        if cached_outer_blocks:
            logger.debug("[PFX] [CACHE-HIT] REQUEST=%s IB=%s -> OB=%s",
                        request_id, cached_ib, cached_outer_blocks)
            return torch.tensor(cached_outer_blocks, dtype=torch.int32)
        return None

    def get_blocks(self, request_id: str) -> torch.Tensor:
        """
        Get all the outer blocks allocated to the given request.
        """
        self.pooled_tensor.fill_(-1)
        value = torch.tensor(self.req_to_outer_blocks[request_id],
                             dtype=torch.int32)
        self.pooled_tensor[:len(value)].copy_(value)
        ret = self.pooled_tensor.clone()
        return ret
