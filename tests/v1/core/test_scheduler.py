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

from typing import Optional

import pytest
from vllm.v1.outputs import ModelRunnerOutput
from vllm.v1.request import RequestStatus

from .utils import create_requests, create_scheduler


def test_add_requests():
    scheduler = create_scheduler()
    requests = create_requests(num_requests=10)

    for i, request in enumerate(requests):
        scheduler.add_request(request)
        assert request.request_id in scheduler.requests
        assert len(scheduler.waiting) == i + 1


def test_finish_request():
    scheduler = create_scheduler()
    requests = create_requests(num_requests=10)
    for request in requests:
        scheduler.add_request(request)

    for i, request in enumerate(requests):
        scheduler.finish_requests(request.request_id,
                                  RequestStatus.FINISHED_ABORTED)
        assert request.request_id not in scheduler.requests
        assert len(scheduler.waiting) == 9 - i


def test_get_num_unfinished_requests():
    scheduler = create_scheduler()
    requests = create_requests(num_requests=10)
    for request in requests:
        scheduler.add_request(request)

    for i, request in enumerate(requests):
        scheduler.finish_requests(request.request_id,
                                  RequestStatus.FINISHED_STOPPED)
        assert scheduler.get_num_unfinished_requests() == len(requests) - i - 1


@pytest.mark.parametrize(
    "enable_prefix_caching, prompt_logprobs",
    [
        (None, None),
    ],
)
def test_schedule_single_seq(enable_prefix_caching: Optional[bool],
                             prompt_logprobs: Optional[int]):
    """Test scheduling.
    Only one case: default APC/no prompt logprobs
    """
    scheduler = create_scheduler(enable_prefix_caching=enable_prefix_caching,
                                 max_num_seqs=1)
    requests = create_requests(num_requests=8,
                               num_tokens=16,
                               prompt_logprobs=prompt_logprobs)

    # Add to Waiting Queue
    for request in requests:
        scheduler.add_request(request)

    # Test initial scheduling
    output = scheduler.schedule()
    assert len(output.scheduled_new_reqs) == 1
    assert len(output.scheduled_cached_reqs) == 0
    assert len(output.finished_req_ids) == 0

    # Verify all tokens in the request are scheduled.
    for req_id, num_tokens in output.num_scheduled_tokens.items():
        assert num_tokens == len(requests[int(req_id)].prompt_token_ids)

    # Verify requests moved from waiting to running
    assert len(scheduler.waiting) == len(requests) - 1
    assert len(scheduler.running) == 1
    assert requests[0] == scheduler.running[0]


@pytest.mark.parametrize(
    "enable_prefix_caching, prompt_logprobs",
    [
        (None, None),
    ],
)
def test_schedule_multi_seq(enable_prefix_caching: Optional[bool],
                            prompt_logprobs: Optional[int]):
    """Test scheduling.
    Only one case: default APC/no prompt logprobs
    """
    scheduler = create_scheduler(enable_prefix_caching=enable_prefix_caching,
                                 max_num_seqs=2)
    requests = create_requests(num_requests=2, prompt_logprobs=prompt_logprobs)

    # Add to Waiting Queue
    for request in requests:
        scheduler.add_request(request)

    # Test initial scheduling
    for _ in range(scheduler.max_num_running_reqs):
        output = scheduler.schedule()
        assert len(output.scheduled_new_reqs) == 1
        assert len(output.scheduled_cached_reqs) == 0
        assert len(output.finished_req_ids) == 0

    # Verify all tokens in the request are scheduled.
    for req_id, num_tokens in output.num_scheduled_tokens.items():
        assert num_tokens == len(requests[int(req_id)].prompt_token_ids)

    # Verify requests moved from waiting to running
    assert (len(scheduler.waiting) == len(requests) -
            scheduler.max_num_running_reqs)
    assert len(scheduler.running) == scheduler.max_num_running_reqs

    for i, running_request in enumerate(scheduler.running):
        assert requests[i] == running_request


@pytest.mark.parametrize(
    "enable_prefix_caching, prompt_logprobs, max_num_seqs, block_size, \
    max_num_batched_tokens, max_model_len, num_blocks, num_tokens_per_batch",
    [
        (False, None, 2, 16, 32, 64, 8, 32),
    ],
)
def test_schedule_alloc_block(
    enable_prefix_caching: Optional[bool],
    prompt_logprobs: Optional[int],
    max_num_seqs: Optional[int],
    block_size: Optional[int],
    max_num_batched_tokens: Optional[int],
    max_model_len: Optional[int],
    num_blocks: Optional[int],
    num_tokens_per_batch: Optional[int],
):
    scheduler = create_scheduler(
        enable_prefix_caching=enable_prefix_caching,
        max_num_seqs=max_num_seqs,
        block_size=block_size,
        max_num_batched_tokens=max_num_batched_tokens,
        max_model_len=max_model_len,
        num_blocks=num_blocks,
    )
    requests = create_requests(
        num_requests=max_num_seqs,
        num_tokens=num_tokens_per_batch,
        prompt_logprobs=prompt_logprobs,
    )

    # Schedule the first request.
    scheduler.add_request(requests[0])
    scheduler_output0 = scheduler.schedule()
    assert len(scheduler_output0.scheduled_new_reqs) == 1
    assert (scheduler_output0.num_scheduled_tokens[requests[0].request_id] ==
            num_tokens_per_batch)
    assert scheduler_output0.scheduled_new_reqs[0].block_ids[0] == [1, 2]

    # Schedule the second request.
    scheduler.add_request(requests[1])
    scheduler_output1 = scheduler.schedule()
    assert len(scheduler_output1.scheduled_new_reqs) == 1
    assert (scheduler_output1.num_scheduled_tokens[requests[1].request_id] ==
            num_tokens_per_batch)
    assert scheduler_output1.scheduled_new_reqs[0].block_ids[0] == [3, 4]

    # Model output of the first request.
    model_runner_output = ModelRunnerOutput(
        req_ids=[requests[0].request_id],
        req_id_to_index={requests[0].request_id: 0},
        sampled_token_ids=[[0]],
        spec_token_ids=None,
        logprobs=None,
        prompt_logprobs_dict={},
    )
    # first request status update
    scheduler.update_from_output(scheduler_output0, model_runner_output)

    # Model output of the second request.
    model_runner_output = ModelRunnerOutput(
        req_ids=[requests[1].request_id],
        req_id_to_index={requests[1].request_id: 0},
        sampled_token_ids=[[0]],
        spec_token_ids=None,
        logprobs=None,
        prompt_logprobs_dict={},
    )
    # second request status update
    scheduler.update_from_output(scheduler_output1, model_runner_output)

    # Schedule the next step again. The first request and second request
    # can be scheduled with decode phase.
    scheduler_output2 = scheduler.schedule()
    assert len(scheduler_output2.num_scheduled_tokens.keys()) == 2
    assert (scheduler_output2.num_scheduled_tokens[requests[0].request_id] == 1
            and scheduler_output2.num_scheduled_tokens[requests[1].request_id]
            == 1)
    assert scheduler_output2.scheduled_cached_reqs[0].new_block_ids[0] == [
        5
    ] and scheduler_output2.scheduled_cached_reqs[1].new_block_ids[0] == [6]


@pytest.mark.parametrize(
    "enable_prefix_caching, prompt_logprobs, max_num_seqs, block_size, \
    max_num_batched_tokens, max_model_len, num_blocks, num_tokens_per_batch",
    [
        (False, None, 2, 16, 32, 64, 5, 32),
    ],
)
def test_schedule_preempted_block(
    enable_prefix_caching: Optional[bool],
    prompt_logprobs: Optional[int],
    max_num_seqs: Optional[int],
    block_size: Optional[int],
    max_num_batched_tokens: Optional[int],
    max_model_len: Optional[int],
    num_blocks: Optional[int],
    num_tokens_per_batch: Optional[int],
):
    scheduler = create_scheduler(
        enable_prefix_caching=enable_prefix_caching,
        max_num_seqs=max_num_seqs,
        block_size=block_size,
        max_num_batched_tokens=max_num_batched_tokens,
        max_model_len=max_model_len,
        num_blocks=num_blocks,
    )
    requests = create_requests(
        num_requests=max_num_seqs,
        num_tokens=num_tokens_per_batch,
        prompt_logprobs=prompt_logprobs,
    )

    # Schedule the first request.
    scheduler.add_request(requests[0])
    scheduler_output0 = scheduler.schedule()
    assert len(scheduler_output0.scheduled_new_reqs) == 1
    assert (scheduler_output0.num_scheduled_tokens[requests[0].request_id] ==
            num_tokens_per_batch)
    assert scheduler_output0.scheduled_new_reqs[0].block_ids[0] == [1, 2]

    # Schedule the second request.
    scheduler.add_request(requests[1])
    scheduler_output1 = scheduler.schedule()
    assert len(scheduler_output1.scheduled_new_reqs) == 1
    assert (scheduler_output1.num_scheduled_tokens[requests[1].request_id] ==
            num_tokens_per_batch)
    assert scheduler_output1.scheduled_new_reqs[0].block_ids[0] == [3, 4]

    # Model output of the first request.
    model_runner_output = ModelRunnerOutput(
        req_ids=[requests[0].request_id],
        req_id_to_index={requests[0].request_id: 0},
        sampled_token_ids=[[0]],
        spec_token_ids=None,
        logprobs=None,
        prompt_logprobs_dict={},
    )
    # first request status update
    scheduler.update_from_output(scheduler_output0, model_runner_output)

    # Model output of the second request.
    model_runner_output = ModelRunnerOutput(
        req_ids=[requests[1].request_id],
        req_id_to_index={requests[1].request_id: 0},
        sampled_token_ids=[[0]],
        spec_token_ids=None,
        logprobs=None,
        prompt_logprobs_dict={},
    )
    # second request status update
    scheduler.update_from_output(scheduler_output1, model_runner_output)

    # Schedule the next step again. The first request and second request
    # can be scheduled with decode phase. But This will cause the
    # preemption of the second request because the KV cache is full.
    scheduler_output2 = scheduler.schedule()
    assert len(scheduler.waiting) == 1, len(scheduler.running) == 1
    assert requests[1].status == RequestStatus.PREEMPTED
    assert scheduler.running[0] == requests[0]
    assert scheduler_output2.num_scheduled_tokens[requests[0].request_id] == 1
    assert scheduler_output2.scheduled_cached_reqs[0].new_block_ids[0] == [
        4
    ]  # preempted_block
