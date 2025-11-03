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
from vllm import SamplingParams
from vllm.platforms import current_platform
from vllm.utils import sha256
from vllm.v1.core.kv_cache_utils import (get_request_block_hasher,
                                         init_none_hash)
from vllm.v1.request import Request

from .utils import create_model_runner_output, create_scheduler

MAX_NUM_SEQ = 2
MAX_MODEL_LEN = 64
OB_SIZE = 16
IB_SIZE = 4
NUM_BLOCKS = MAX_MODEL_LEN // IB_SIZE * MAX_NUM_SEQ + 1  # 9
DEVICE = current_platform.device_type
HASH_FN = sha256

# checkout output scheduler, runner in prefill, decode tests
# not preempt + eviction test
# preempt + eviction test


@pytest.fixture
def scheduler():
    scheduler = create_scheduler(
        max_num_seqs=MAX_NUM_SEQ,
        max_num_batched_tokens=MAX_MODEL_LEN,
        num_blocks=NUM_BLOCKS,
        block_size=IB_SIZE,
        max_model_len=MAX_MODEL_LEN,
        outer_block_size=OB_SIZE,
        enable_prefix_caching=True,
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
        pooling_params=None,
        eos_token_id=100,
    )
    return request


@pytest.mark.parametrize(
    "token_length, cached_block_table, cached_length, ",
    [
        pytest.param(
            50,
            [0, 1, 2],
            [16, 16, 16],
            id="50_tokens",
        ),
        pytest.param(
            54,
            [0, 1, 2, 3],
            [16, 16, 16, 4],
            id="54_tokens",
        ),
    ],
)
def test_prefix_cache_hit_same_prompt(scheduler, token_length: int,
                                      cached_block_table: list[int],
                                      cached_length: list[int]):
    init_none_hash(HASH_FN)
    """
    Check the prefix caching works as expected
    between two requests with the same prompt.
    Note that if the prompt is fully cached,
    the last inner block is excluded from cache hit
    to return the logits.
    """
    # 1. Generate req0: 42 tokens -> 11 inner blocks
    common_token_ids = [i for i in range(token_length)]
    req0_id = "req0"
    req0 = create_request(req0_id, common_token_ids, IB_SIZE, HASH_FN)

    req1_id = "req1"
    req1 = create_request(req1_id, common_token_ids, IB_SIZE, HASH_FN)

    requests = [req0, req1]
    for request in requests:
        scheduler.add_request(request)

    output = scheduler.schedule()
    answer = torch.tensor([0, 1, 2, 3], dtype=torch.int32)
    assert torch.allclose(output.block_table_dict[req0_id], answer)
    assert output.cached_block_table == []
    assert output.cached_length == []

    # Model output of the first request.
    model_runner_output = create_model_runner_output(output)
    scheduler.update_from_output(output, model_runner_output)

    output = scheduler.schedule()
    answer = torch.tensor([4, 5, 6, 7], dtype=torch.int32)
    assert torch.allclose(output.block_table_dict[req1_id], answer)
    assert output.cached_block_table == cached_block_table
    assert output.cached_length == cached_length
