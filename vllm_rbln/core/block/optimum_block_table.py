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

import math
from typing import List, Optional

from vllm.core.block.block_table import BlockTable
from vllm.core.block.common import BlockList
from vllm.core.block.interfaces import Block, DeviceAwareBlockAllocator


class RBLNOptimumBlockTable(BlockTable):

    def __init__(
        self,
        block_size: int,
        block_allocator: DeviceAwareBlockAllocator,
        _blocks: Optional[List[Block]] = None,
        max_block_sliding_window: Optional[int] = None,
    ):
        self._block_size = block_size
        self._allocator = block_allocator
        if _blocks is None:
            _blocks = []
        self._blocks: BlockList = BlockList(_blocks)
        # NOTE We manage sliding window attention
        # outside of the core block table.
        self._max_block_sliding_window = None
        self._num_full_slots = self._get_num_token_ids()

    def get_num_blocks_touched_by_append_slots(
            self, token_ids: List[int], num_lookahead_slots: int) -> int:
        """Determine how many blocks will be "touched" by appending the token
        ids.

        This is required for the scheduler to determine whether a sequence can
        continue generation, or if it must be preempted.
        """
        # Math below is equivalent to:
        # all_token_ids = token_ids + [-1] * num_lookahead_slots
        # token_blocks = self._chunk_token_blocks_for_append(all_token_ids)
        # return len(token_blocks)

        num_token_ids = len(token_ids) + num_lookahead_slots
        first_chunk_size = self._block_size - (self._num_full_slots %
                                               self._block_size)
        if first_chunk_size == self._block_size:
            return math.ceil(num_token_ids / self._block_size)

        return math.ceil((num_token_ids - first_chunk_size) / self._block_size)
