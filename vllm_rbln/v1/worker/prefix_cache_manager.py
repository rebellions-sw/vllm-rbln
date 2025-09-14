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

    def get_num_free_blocks(self) -> int:
        return len(self.blocks)


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
        # TODO Save only newly allocated inner blocks
        self.inner_to_outer_block: dict[int, list[int]] = {}
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

    def _evict_cached_ib(self, num_evicted: int) -> None:
        """
        Evict the outer blocks of finished requests
        as much as num_evicted
        in a round-robin manner.
        """
        evicted = 0
        for ob_id, (ob, ibs) in enumerate(self.outer_to_inner_blocks):
            if evicted >= num_evicted:
                break
            if len(ibs) == 0:
                continue
            if ob_id not in self.outer_block_to_req:
                self.outer_block_manager.append(ob)
                logger.debug("[PFX] [EVICTION] OB=%d (IB=%s)", ob_id, ibs)
                for ib in ibs:
                    self.inner_to_outer_block[ib].remove(ob_id)
                self.outer_to_inner_blocks[ob_id] = tuple()
                evicted += 1

    def _allocate_new_ob(self, num_obs: int = 1) -> int:
        """
        Allocate new outer blocks for the given number of outer blocks.
        If no free outer blocks, evict the outer blocks of finished requests.
        """
        num_evicted = num_obs - self.outer_block_manager.get_num_free_blocks()
        if num_evicted > 0:
            self._evict_cached_ib(num_evicted)
        new_obs = [self.outer_block_manager.popleft() for _ in range(num_obs)]
        return new_obs

    def _allocate_ibs_per_ob(self, new_ob: RBLNBlock,
                             inner_blocks: list[int]) -> None:
        for ib_id in inner_blocks:
            if ib_id not in self.inner_to_outer_block:
                self.inner_to_outer_block[ib_id] = []
            self.inner_to_outer_block[ib_id].append(new_ob.block_id)

        self.outer_to_inner_blocks[new_ob.block_id] = (new_ob, inner_blocks)

    def _append_new_ib(self, ob_id: int, inner_blocks: list[int]) -> None:
        """
        Append new inner blocks to the last outer block of the request.
        """
        assert len(inner_blocks) == 1
        new_ib = inner_blocks[0]
        if new_ib not in self.inner_to_outer_block:
            self.inner_to_outer_block[new_ib] = []
        self.inner_to_outer_block[new_ib].append(ob_id)
        self.outer_to_inner_blocks[ob_id][1].append(new_ib)

    def allocate_blocks(self, request_id: str, cached_len: int,
                        inner_blocks: list[int]) -> None:
        """
        Allocate outer blocks for the given inner blocks.
        """
        if request_id in self.req_to_outer_blocks:
            # DECODE
            num_already_allocated_ibs = cached_len // self.ib_size
            if num_already_allocated_ibs % self.blk_ratio == 0:
                num_obs = 1
            else:
                last_ob_id = self.req_to_outer_blocks[request_id][-1]
                self._append_new_ib(last_ob_id, inner_blocks)
                return
        else:
            # PREFILL
            self.req_to_outer_blocks[request_id] = []
            num_obs = (len(inner_blocks) + self.blk_ratio -
                       1) // self.blk_ratio

        new_obs = self._allocate_new_ob(num_obs)
        for ob_idx, new_ob in enumerate(new_obs):
            # Map the new outer block to the request.
            self.req_to_outer_blocks[request_id].append(new_ob.block_id)
            self.outer_block_to_req[new_ob.block_id] = request_id

            # Map the new outer block to the inner blocks.
            start_pos = ob_idx * self.blk_ratio
            end_pos = min((ob_idx + 1) * self.blk_ratio, len(inner_blocks))
            self._allocate_ibs_per_ob(new_ob, inner_blocks[start_pos:end_pos])

        obs = self.req_to_outer_blocks[request_id]
        logger.debug("[PFX] [ALLOC] REQUEST=%s OB=%s (IB=%s)", request_id, obs,
                     inner_blocks)

    def free_request(self, request_id: str) -> None:
        """
        Remove the mapping of the given request.
        Keep the outer blocks alive until their inner blocks
        are reused by other requests.
        """
        outer_blocks = self.req_to_outer_blocks.pop(request_id, None)
        for ob_id in outer_blocks:
            self.outer_block_to_req.pop(ob_id, None)

    def get_cached_origin_blocks(
            self, request_id, cached_len,
            inner_blocks: list[int]) -> tuple[torch.Tensor, int]:
        """
        Get the request ID that is cached and the length of cached tokens.
        """
        cached_ob = []
        cached_request_id = None
        real_cached_len = 0
        num_cached_ib = self._num_cached_inner_blocks(cached_len)
        # NOTE blocks in cached_ib are all cached or not.
        cached_ib = inner_blocks[:num_cached_ib]
        for req_id, obs in self.req_to_outer_blocks.items():
            if request_id == req_id:
                continue
            cur_cached_ob = []
            for ob_idx, ob_id in enumerate(obs):
                # TODO assert
                start_pos = ob_idx * self.blk_ratio
                end_pos = min((ob_idx + 1) * self.blk_ratio, len(cached_ib))
                if start_pos >= len(cached_ib):
                    break
                cur_cached_ib = cached_ib[start_pos:end_pos]
                ob, ibs = self.outer_to_inner_blocks[ob_id]
                if len(ibs) < len(cur_cached_ib):
                    break
                if ibs[:len(cur_cached_ib)] == cur_cached_ib:
                    cur_cached_ob.append(ob_id)
                    real_cached_len += len(cur_cached_ib) * self.ib_size
            if len(cur_cached_ob) > len(cached_ob):
                cached_ob = cur_cached_ob
                cached_request_id = req_id

        if cached_request_id is not None:
            logger.debug(
                "[PFX] [CACHE-HIT] REQUEST=%s -> REQUEST=%s (IB=%s of OB=%s)",
                request_id, cached_request_id, cached_ib, cached_ob)
            cached_ob_tensor = torch.tensor(cached_ob, dtype=torch.int32)
        else:
            cached_ob_tensor = None
            real_cached_len = 0
        return cached_ob_tensor, real_cached_len

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
