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

import pytest
import torch
from vllm.platforms import current_platform
from vllm.v1.core.kv_cache_manager import KVCacheManager

from vllm_rbln.v1.worker.optimum_model_runner import RBLNOptimumModelRunner
from vllm_rbln.v1.worker.prefix_cache_manager import RBLNPrefixKVCacheManager

from .utils import (MockModelWrapper, _schedule_cached_reqs,
                    _schedule_new_request, finish_request, get_vllm_config,
                    initialize_kv_cache, make_kv_cache_config, make_request)

MAX_NUM_SEQ = 2
MAX_MODEL_LEN = 64
OB_SIZE = 16
IB_SIZE = 4
NUM_BLOCKS = MAX_MODEL_LEN // OB_SIZE * MAX_NUM_SEQ + 1
DEVICE = current_platform.device_type


@pytest.fixture
def model_runner():
    vllm_config = get_vllm_config()

    runner = RBLNOptimumModelRunner(vllm_config, DEVICE)
    runner.model = MockModelWrapper()
    runner.prefix_cache_manager = RBLNPrefixKVCacheManager(
        ob_size=OB_SIZE,
        ib_size=IB_SIZE,
        max_model_len=MAX_MODEL_LEN,
        num_ob=runner.model.model.kv_block_adapter.get_available_num_blocks(),
    )
    initialize_kv_cache(runner)
    return runner


def test_prefill(model_runner):
    """
    Check the prefix caching works as expected during prefill.

    req0: 42 tokens -> 11 inner blocks -> 3 outer blocks allocated
    req1: 42 (32 + 10) tokens
        -> 8 cached + 3 new inner blocks allocated
        -> 3 outer blocks allocated
    req0 finished and freed
    req2: 50 (20 + 30) tokens
        -> 5 cached + 8 new inner blocks allocated
        -> 13 outer blocks allocated
    """
    manager = KVCacheManager(
        make_kv_cache_config(
            block_size=IB_SIZE,
            num_blocks=NUM_BLOCKS * (OB_SIZE // IB_SIZE),
        ),
        max_model_len=MAX_MODEL_LEN,
        enable_caching=True,
    )

    # 1. Generate req0: 42 tokens -> 11 inner blocks
    common_token_ids = [i for i in range(32)]
    unique_token_ids = [3] * 10
    all_token_ids = common_token_ids + unique_token_ids
    req_id = "0"
    req0 = make_request(req_id, all_token_ids)
    computed_blocks, num_computed_tokens = manager.get_computed_blocks(req0)
    assert not computed_blocks.blocks[0]
    assert num_computed_tokens == 0
    blocks = manager.allocate_slots(req0, len(all_token_ids),
                                    len(computed_blocks.blocks[0]) * IB_SIZE,
                                    computed_blocks)
    assert blocks.get_block_ids() == ([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11], )
    # Check the allocated blocks
    scheduler_output = _schedule_new_request(
        req_id,
        token_ids=all_token_ids,
        block_ids=blocks.get_block_ids(),
        new_computed_tokens=num_computed_tokens)
    model_runner._update_states(scheduler_output)
    inputs = model_runner._prepare_inputs(scheduler_output)
    assert torch.allclose(inputs.block_tables[0],
                          torch.tensor([0, 1, 2, -1], dtype=torch.int32))
    assert inputs.cached_block_tables is None

    # 2. Generate partially cached request req1
    unique_token_ids = [1] * 10
    all_token_ids = common_token_ids + unique_token_ids
    req_id = "1"
    req1 = make_request(req_id, all_token_ids)
    computed_blocks, num_computed_tokens = manager.get_computed_blocks(req1)
    assert len(computed_blocks.blocks[0]) == 8
    assert num_computed_tokens == 32
    blocks = manager.allocate_slots(req1, len(unique_token_ids),
                                    len(computed_blocks.blocks[0]) * IB_SIZE,
                                    computed_blocks)
    # [1, 2, 3, 4, 5, 6, 7, 8] are cached
    # Allocate [12, 13, 14] for new 3 inner blocks
    assert blocks.get_block_ids() == ([12, 13, 14], )
    total_allocated_blocks = manager.get_block_ids(req1.request_id)
    assert total_allocated_blocks == ([1, 2, 3, 4, 5, 6, 7, 8, 12, 13, 14], )
    scheduler_output = _schedule_new_request(
        req_id,
        token_ids=all_token_ids,
        block_ids=total_allocated_blocks,
        new_computed_tokens=num_computed_tokens)
    model_runner._update_states(scheduler_output)
    inputs = model_runner._prepare_inputs(scheduler_output)

    assert torch.allclose(inputs.block_tables[0],
                          torch.tensor([3, 4, 5, -1], dtype=torch.int32))
    assert torch.allclose(inputs.cached_block_tables,
                          torch.tensor([[0, 1]], dtype=torch.int32))
    # 3. Finish req1 and schedule req2
    finished_req = req1
    finish_request(manager, finished_req)

    # Allocate 13 inner blocks (50 tokens) for req2
    req_id = "2"
    common_token_ids = [i for i in range(20)]
    unique_token_ids = [2] * 30
    all_token_ids = common_token_ids + unique_token_ids
    req2 = make_request(req_id, all_token_ids)
    computed_blocks, num_computed_tokens = manager.get_computed_blocks(req2)
    assert len(computed_blocks.blocks[0]) == 5
    assert num_computed_tokens == 20
    blocks = manager.allocate_slots(req2, len(unique_token_ids),
                                    len(computed_blocks.blocks[0]) * IB_SIZE,
                                    computed_blocks)
    # [1, 2, 3, 4, 5] are cached
    # Allocate [15, 16, 17, 18, 19, 20, 21, 22] for new 8 inner blocks
    assert blocks.get_block_ids() == ([15, 16, 17, 18, 19, 20, 21, 22], )
    total_allocated_blocks = manager.get_block_ids(req2.request_id)
    assert total_allocated_blocks == ([
        1, 2, 3, 4, 5, 15, 16, 17, 18, 19, 20, 21, 22
    ], )
    scheduler_output = _schedule_new_request(
        req_id,
        token_ids=all_token_ids,
        block_ids=total_allocated_blocks,
        new_computed_tokens=num_computed_tokens,
        finished_req_ids=[finished_req.request_id])
    model_runner._update_states(scheduler_output)
    inputs = model_runner._prepare_inputs(scheduler_output)

    # Check the allocated outer blocks
    assert torch.allclose(inputs.block_tables[0],
                          torch.tensor([6, 7, 8, -1], dtype=torch.int32))
    assert torch.allclose(inputs.cached_block_tables,
                          torch.tensor([[0, 1]], dtype=torch.int32))


@pytest.mark.parametrize(
    "num_generated_token_ids, new_inner_blocks, outer_blocks_allocated",
    [
        pytest.param(4, [3], [0, -1, -1, -1],
                     id="without-new-outer-block-allocated"),
        pytest.param(30, [3, 4, 5, 6, 7, 8, 9], [0, 1, 2, -1],
                     id="with-new-outer-block-allocated"),
    ],
)
def test_decode(model_runner, num_generated_token_ids, new_inner_blocks,
                outer_blocks_allocated):
    """
    Check the prefix caching works as expected during decode.
    """

    manager = KVCacheManager(
        make_kv_cache_config(
            block_size=IB_SIZE,
            num_blocks=NUM_BLOCKS * (OB_SIZE // IB_SIZE),
        ),
        max_model_len=MAX_MODEL_LEN,
        enable_caching=True,
    )

    # 1. Generate req0: 6 tokens -> 2 inner blocks
    common_token_ids = [i for i in range(4)]
    unique_token_ids = [3] * 2

    all_token_ids = common_token_ids + unique_token_ids
    req_id = "0"
    req0 = make_request(req_id, all_token_ids)
    computed_blocks, num_computed_tokens = manager.get_computed_blocks(req0)
    assert not computed_blocks.blocks[0]
    assert num_computed_tokens == 0
    blocks = manager.allocate_slots(req0, len(all_token_ids),
                                    len(computed_blocks.blocks[0]) * IB_SIZE,
                                    computed_blocks)
    assert blocks.get_block_ids() == ([1, 2], )
    scheduler_output = _schedule_new_request(
        req_id,
        token_ids=all_token_ids,
        block_ids=blocks.get_block_ids(),
        new_computed_tokens=num_computed_tokens)
    model_runner._update_states(scheduler_output)
    inputs = model_runner._prepare_inputs(scheduler_output)
    assert torch.allclose(inputs.block_tables[0],
                          torch.tensor([0, -1, -1, -1], dtype=torch.int32))
    assert inputs.cached_block_tables is None

    # 2. Decode req0: `num_generated_token_ids` tokens
    req0.num_computed_tokens = len(all_token_ids)
    for _ in range(num_generated_token_ids):
        req0.append_output_token_ids(1)

    blocks = manager.allocate_slots(req0, num_generated_token_ids)
    assert blocks.get_block_ids() == (new_inner_blocks, )
    scheduler_output = _schedule_cached_reqs(
        [req0],
        [blocks.get_block_ids()],
    )
    model_runner._update_states(scheduler_output)
    inputs = model_runner._prepare_inputs(scheduler_output)
    assert torch.allclose(
        inputs.block_tables[0],
        torch.tensor(outer_blocks_allocated, dtype=torch.int32))
    assert inputs.cached_block_tables is None
