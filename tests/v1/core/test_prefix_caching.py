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
from vllm.utils import sha256
from vllm.v1.core.kv_cache_manager import KVCacheManager
from vllm.v1.core.kv_cache_utils import init_none_hash

from vllm_rbln.prefix_cache_manager.optimum_prefix_cache_manager import (
    RBLNPrefixKVCacheManager)

from .utils import (_schedule_cached_reqs, _schedule_new_request,
                    finish_request, make_kv_cache_config, make_request)

MAX_NUM_SEQ = 2
MAX_MODEL_LEN = 64
OB_SIZE = 16
IB_SIZE = 4
NUM_BLOCKS = MAX_MODEL_LEN // OB_SIZE * MAX_NUM_SEQ + 1  # 9
DEVICE = current_platform.device_type
HASH_FN = sha256

# checkout output scheduler, runner in prefill, decode tests
# not preempt + eviction test
# preempt + eviction test

@pytest.fixture
def scheduler():
    create_scheduler(
        max_num_seqs=MAX_NUM_SEQ,
        max_num_batched_tokens=MAX_MODEL_LEN,
        num_blocks=NUM_BLOCKS,
        block_size=IB_SIZE,
        max_model_len=MAX_MODEL_LEN,
        outer_block_size=OB_SIZE,
    )
    return scheduler

def create_request(
    request_id: str,
    prompt_token_ids: list[int],
    block_size: int,
    hash_fn: callable,
) -> Request:
    block_hasher = get_request_block_hasher(block_size, hash_fn)
    request = Request(
        request_id=request_id,
        prompt_token_ids=prompt_token_ids,
        sampling_params=SamplingParams(
            max_tokens=MAX_MODEL_LEN,
            temperature=0.0,
        ),
        block_hasher=block_hasher,
    )
    return request

def test_prefix_cache_hit_same_prompt():
    init_none_hash(HASH_FN)
    """
    Check the prefix caching works as expected
    between two requests with the same prompt.

    req0: 50 tokens -> 13 inner blocks -> 4 outer blocks allocated
    req1: same 50 tokens
        -> 12 cached + 1 new inner blocks allocated
        -> 3 outer blocks cache hit
    """
    # 1. Generate req0: 42 tokens -> 11 inner blocks
    common_token_ids = [i for i in range(50)]
    req0_id = "0"
    req0 = create_request(req0_id, common_token_ids, IB_SIZE, HASH_FN)

    req1_id = "1"
    req1 = create_request(req1_id, common_token_ids, IB_SIZE, HASH_FN)

    requests = [req0, req1]
    for request in requests:
        scheduler.add_request(request)
    
    output = scheduler.schedule()
    assert output.block_table_dict[req0_id] == [0, 1, 2, 3]
    assert output.cached_block_table == []
    assert output.cached_length == []

    # Model output of the first request.
    model_runner_output = create_model_runner_output(output)
    scheduler.update_from_output(output, model_runner_output)

    output = scheduler.schedule()
    assert output.block_table_dict[req1_id] == [4, 5, 6, 7]
    assert output.cached_block_table == [0, 1, 2]
    assert output.cached_length == [16, 16, 16, 4]


def test_prefill(scheduler, runner):
    init_none_hash(HASH_FN)
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
    # 1. Generate req0: 42 tokens -> 11 inner blocks
    common_token_ids = [i for i in range(32)]
    req0_id = "0"
    unique_token_ids = [3] * 10
    all_token_ids = common_token_ids + unique_token_ids
    req0 = create_request(req0_id, all_token_ids, IB_SIZE, HASH_FN)

    req1_id = "1"
    unique_token_ids = [1] * 10
    all_token_ids = common_token_ids + unique_token_ids
    req1 = create_request(req1_id, all_token_ids, IB_SIZE, HASH_FN)

    req2_id = "2"
    unique_token_ids = [2] * 30
    all_token_ids = common_token_ids + unique_token_ids
    req2 = create_request(req2_id, all_token_ids, IB_SIZE, HASH_FN)
    requests = [req0, req1, req2]


    for request in requests:
        scheduler.add_request(request)
    
    output = scheduler.schedule()
    assert output.block_table_dict[req0_id] == [0, 1, 2, -1]
    assert output.cached_block_table == []
    assert output.cached_length == []
    # Check model input in runner will be checked in worker directory.

    # Model output of the first request.
    model_runner_output = create_model_runner_output(output)
    scheduler.update_from_output(output, model_runner_output)

    output = scheduler.schedule()
    assert output.block_table_dict[req1_id] == [3, 4, 5, -1]
    assert output.cached_block_table == [0, 1]
    assesrt output.cached_length == [16, 16]

    # Model output of the second request.
    model_runner_output = create_model_runner_output(output)
    scheduler.update_from_output(output, model_runner_output)







    # computed_blocks, num_computed_tokens = kv_cache_manager.get_computed_blocks(req0)
    # assert not computed_blocks.blocks[0]
    # assert num_computed_tokens == 0
    # blocks = kv_cache_manager.allocate_slots(req0, len(all_token_ids), 0,
    #                                 computed_blocks)
    # assert blocks.get_block_ids() == ([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11], )
    # # Check the allocated blocks
    # scheduler_output = _schedule_new_request(
    #     req_id,
    #     token_ids=all_token_ids,
    #     block_ids=blocks.get_block_ids(),
    #     new_computed_tokens=num_computed_tokens,
    #     new_computed_blocks=computed_blocks.get_block_ids()[0],
    # )
    # model_runner._update_states(scheduler_output)
    # inputs, _ = model_runner._prepare_inputs(scheduler_output)
    # assert torch.allclose(inputs.block_tables[0],
    #                       torch.tensor([0, 1, 2, -1], dtype=torch.int32))
    # assert inputs.cached_block_tables == []

    # # 2. Generate partially cached request req1
    # unique_token_ids = [1] * 10
    # all_token_ids = common_token_ids + unique_token_ids
    # req_id = "1"
    # req1 = make_request(req_id, all_token_ids, IB_SIZE, HASH_FN)
    # computed_blocks, num_computed_tokens = manager.get_computed_blocks(req1)
    # assert len(computed_blocks.blocks[0]) == 8
    # assert num_computed_tokens == 32
    # blocks = manager.allocate_slots(req1, len(unique_token_ids),
    #                                 len(computed_blocks.blocks[0]) * IB_SIZE,
    #                                 computed_blocks)
    # # [1, 2, 3, 4, 5, 6, 7, 8] are cached
    # # Allocate [12, 13, 14] for new 3 inner blocks
    # assert blocks.get_block_ids() == ([12, 13, 14], )
    # total_allocated_blocks = manager.get_block_ids(req1.request_id)
    # assert total_allocated_blocks == ([1, 2, 3, 4, 5, 6, 7, 8, 12, 13, 14], )
    # scheduler_output = _schedule_new_request(
    #     req_id,
    #     token_ids=all_token_ids,
    #     block_ids=total_allocated_blocks,
    #     new_computed_tokens=num_computed_tokens,
    #     new_computed_blocks=computed_blocks.get_block_ids()[0],
    # )
    # model_runner._update_states(scheduler_output)
    # inputs, _ = model_runner._prepare_inputs(scheduler_output)

    # assert torch.allclose(inputs.block_tables[0],
    #                       torch.tensor([3, 4, 5, -1], dtype=torch.int32))
    # assert inputs.cached_block_tables == [0, 1]
    # # 3. Finish req1 and schedule req2
    # finished_req = req1
    # finish_request(manager, finished_req)

    # # Allocate 13 inner blocks (50 tokens) for req2
    # req_id = "2"
    # common_token_ids = [i for i in range(20)]
    # unique_token_ids = [2] * 30
    # all_token_ids = common_token_ids + unique_token_ids
    # req2 = make_request(req_id, all_token_ids, IB_SIZE, HASH_FN)
    # computed_blocks, num_computed_tokens = manager.get_computed_blocks(req2)
    # assert len(computed_blocks.blocks[0]) == 5
    # assert num_computed_tokens == 20
    # blocks = manager.allocate_slots(req2, len(unique_token_ids),
    #                                 len(computed_blocks.blocks[0]) * IB_SIZE,
    #                                 computed_blocks)
    # # [1, 2, 3, 4, 5] are cached
    # # Allocate [15, 16, 17, 18, 19, 20, 21, 22] for new 8 inner blocks
    # assert blocks.get_block_ids() == ([15, 16, 17, 18, 19, 20, 21, 22], )
    # total_allocated_blocks = manager.get_block_ids(req2.request_id)
    # assert total_allocated_blocks == ([
    #     1, 2, 3, 4, 5, 15, 16, 17, 18, 19, 20, 21, 22
    # ], )
    # scheduler_output = _schedule_new_request(
    #     req_id,
    #     token_ids=all_token_ids,
    #     block_ids=total_allocated_blocks,
    #     new_computed_tokens=num_computed_tokens,
    #     finished_req_ids=[finished_req.request_id],
    #     new_computed_blocks=computed_blocks.get_block_ids()[0],
    # )
    # model_runner._update_states(scheduler_output)
    # inputs, _ = model_runner._prepare_inputs(scheduler_output)

    # # Check the allocated outer blocks
    # assert torch.allclose(inputs.block_tables[0],
    #                       torch.tensor([6, 7, 8, 3], dtype=torch.int32))
    # assert inputs.cached_block_tables == [0, 1]


def test_decode(model_runner):
    """
    Check the prefix caching works as expected during decode.
    """
    init_none_hash(HASH_FN)
    manager = KVCacheManager(
        make_kv_cache_config(
            block_size=IB_SIZE,
            num_blocks=NUM_BLOCKS * (OB_SIZE // IB_SIZE),
        ),
        max_model_len=MAX_MODEL_LEN,
        enable_caching=True,
    )

    # 1. Generate req0: 16 tokens -> 4 inner blocks
    all_token_ids = [i for i in range(OB_SIZE)]

    req_id = "0"
    req0 = make_request(req_id, all_token_ids, IB_SIZE, HASH_FN)
    computed_blocks, num_computed_tokens = manager.get_computed_blocks(req0)
    assert not computed_blocks.blocks[0]
    assert num_computed_tokens == 0
    blocks = manager.allocate_slots(req0, len(all_token_ids),
                                    len(computed_blocks.blocks[0]) * IB_SIZE,
                                    computed_blocks)
    assert blocks.get_block_ids() == ([1, 2, 3, 4], )
    scheduler_output = _schedule_new_request(
        req_id,
        token_ids=all_token_ids,
        block_ids=blocks.get_block_ids(),
        new_computed_tokens=num_computed_tokens)
    model_runner._update_states(scheduler_output)
    inputs, _ = model_runner._prepare_inputs(scheduler_output)
    assert torch.allclose(inputs.block_tables[0],
                          torch.tensor([0, -1, -1, -1], dtype=torch.int32))
    assert inputs.cached_block_tables == []

    # 2. Decode req0: 1 new token
    # -> 1 new inner block -> 1 new outer block allocated
    req0.num_computed_tokens = len(all_token_ids)
    req0.append_output_token_ids(1)
    new_inner_blocks = [5]
    num_generated_token_ids = 1
    outer_blocks_allocated = [0, 1, -1, -1]

    blocks = manager.allocate_slots(req0, num_generated_token_ids)
    assert blocks.get_block_ids() == (new_inner_blocks, )
    scheduler_output = _schedule_cached_reqs(
        [req0],
        [blocks.get_block_ids()],
    )
    model_runner._update_states(scheduler_output)
    inputs, _ = model_runner._prepare_inputs(scheduler_output)
    assert torch.allclose(
        inputs.block_tables[0],
        torch.tensor(outer_blocks_allocated, dtype=torch.int32))
    assert inputs.cached_block_tables == []


def test_simple_eviction():
    """
    req0: 64 tokens -> 16 inner blocks -> 4 outer blocks allocated
    req1: 64 tokens -> 16 inner blocks -> 4 outer blocks allocated
    finish req0
    req2: 64 tokens -> 16 inner blocks -> 4 outer blocks allocated (evict req0)
    """
    init_none_hash(HASH_FN)
    num_ib = NUM_BLOCKS * (OB_SIZE // IB_SIZE)
    manager = KVCacheManager(
        make_kv_cache_config(
            block_size=IB_SIZE,
            num_blocks=num_ib,
        ),
        max_model_len=MAX_MODEL_LEN,
        enable_caching=True,
    )
    print((manager.block_pool.get_num_free_blocks()), "blocks available")

    prefix_cache_manager = RBLNPrefixKVCacheManager(
        ob_size=OB_SIZE,
        ib_size=IB_SIZE,
        max_model_len=MAX_MODEL_LEN,
        num_ob=NUM_BLOCKS - 1,  # -1 = reserve one outer block for null block
    )

    req_id = "0"
    num_tokens = 64
    all_token_ids = list(range(num_tokens))
    req0 = make_request(req_id, all_token_ids, IB_SIZE, HASH_FN)
    computed_blocks, num_computed_tokens = manager.get_computed_blocks(req0)
    assert not computed_blocks.blocks[0]
    assert num_computed_tokens == 0
    blocks = manager.allocate_slots(req0, len(all_token_ids),
                                    len(computed_blocks.blocks[0]) * IB_SIZE,
                                    computed_blocks)
    golden_inner_block_ids = list(range(1, 17))
    assert blocks.get_block_ids() == (golden_inner_block_ids, )

    obs, _, _ = prefix_cache_manager.get_block_table_prefill(
        req_id,
        cached_blocks=computed_blocks.get_block_ids()[0],
        num_cached_tokens=num_computed_tokens,
        inner_blocks=blocks.get_block_ids()[0],
    )
    assert torch.allclose(obs, torch.tensor([0, 1, 2, 3], dtype=torch.int32))

    req_id = "1"
    all_token_ids = list(
        range(all_token_ids[-1] + 1, all_token_ids[-1] + 1 + num_tokens))
    req1 = make_request(req_id, all_token_ids, IB_SIZE, HASH_FN)
    computed_blocks, num_computed_tokens = manager.get_computed_blocks(req1)
    blocks = manager.allocate_slots(req1, len(all_token_ids),
                                    len(computed_blocks.blocks[0]) * IB_SIZE,
                                    computed_blocks)
    golden_inner_block_ids = list(range(17, 33))
    assert blocks.get_block_ids() == (golden_inner_block_ids, )
    obs, _, _ = prefix_cache_manager.get_block_table_prefill(
        req_id,
        cached_blocks=computed_blocks.get_block_ids()[0],
        num_cached_tokens=num_computed_tokens,
        inner_blocks=blocks.get_block_ids()[0],
    )
    assert torch.allclose(obs, torch.tensor([4, 5, 6, 7], dtype=torch.int32))

    # Finish req0
    manager.free(req0)
    prefix_cache_manager.free_request("0")

    req_id = "2"
    all_token_ids = list(
        range(all_token_ids[-1] + 1, all_token_ids[-1] + 1 + num_tokens))
    req2 = make_request(req_id, all_token_ids, IB_SIZE, HASH_FN)
    computed_blocks, num_computed_tokens = manager.get_computed_blocks(req2)
    blocks = manager.allocate_slots(req2, len(all_token_ids),
                                    len(computed_blocks.blocks[0]) * IB_SIZE,
                                    computed_blocks)
    remained_blocks = list(range(33, num_ib))
    # In vLLM, the blocks are returned
    # to the free block queue in reversed order.
    # It is for preventing memory fragmentation.
    golden_inner_block_ids = remained_blocks + list(
        reversed(range(len(remained_blocks) + 1, 17)))
    assert blocks.get_block_ids() == (golden_inner_block_ids, )
    obs, _, _ = prefix_cache_manager.get_block_table_prefill(
        req_id,
        cached_blocks=computed_blocks.get_block_ids()[0],
        num_cached_tokens=num_computed_tokens,
        inner_blocks=blocks.get_block_ids()[0],
    )
    assert torch.allclose(obs, torch.tensor([0, 1, 2, 3], dtype=torch.int32))


# TODO add preeemption case
