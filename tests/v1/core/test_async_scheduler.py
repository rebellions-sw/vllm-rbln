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
from typing import Optional
from .utils import create_requests, create_scheduler
from vllm.v1.outputs import ModelRunnerOutput
from vllm.v1.request import RequestStatus
from vllm.v1.core.sched.output import SchedulerOutput

def _make_model_runner_output(
    scheduler_output: SchedulerOutput, ) -> ModelRunnerOutput:
    req_ids = list(scheduler_output.num_scheduled_tokens.keys())
    return ModelRunnerOutput(
        req_ids=req_ids,
        req_id_to_index={
            req_id: i
            for i, req_id in enumerate(req_ids)
        },
        sampled_token_ids=[[i] for i in range(len(req_ids))],
        spec_token_ids=None,
        logprobs=None,
        prompt_logprobs_dict={},
    )

@pytest.mark.parametrize(
    "max_num_seqs, block_size, max_model_len, num_blocks, num_tokens_per_batch, "
    "exp_new_req0_blocks, exp_new_req1_blocks, exp_cached0_new, exp_cached1_new",
    [
        pytest.param(
            2, 16, 64, 8, 32, [1, 2], [3, 4], [5], [6],
            id="16bsize-32len"
        ),
        pytest.param(
            4, 8, 24, 7, 17, [1, 2, 3], [4, 5, 6], [], [],
            id="8bsize-17len"
        )
    ],
)
def test_schedule_alloc_block(
    max_num_seqs: Optional[int],
    block_size: Optional[int],
    max_model_len: Optional[int],
    num_blocks: Optional[int],
    num_tokens_per_batch: Optional[int],
    exp_new_req0_blocks: list[int],
    exp_new_req1_blocks: list[int],
    exp_cached0_new: list[int],
    exp_cached1_new: list[int],
):
    scheduler = create_scheduler(
        max_num_seqs=max_num_seqs,
        block_size=block_size,
        max_num_batched_tokens=max_model_len,
        max_model_len=max_model_len,
        num_blocks=num_blocks,
        async_scheduling=True,
    )

    requests = create_requests(
        num_requests=max_num_seqs,
        num_tokens=num_tokens_per_batch,
    )

    # Schedule the first request.
    scheduler.add_request(requests[0])
    scheduler_output0 = scheduler.schedule()
    assert scheduler_output0.scheduled_new_reqs[0].block_ids[0] == exp_new_req0_blocks

    # Schedule the second request.
    scheduler.add_request(requests[1])
    scheduler_output1 = scheduler.schedule()
    assert scheduler_output1.scheduled_new_reqs[0].block_ids[0] == exp_new_req1_blocks

    # Model output of the first request.
    model_runner_output = _make_model_runner_output(scheduler_output0)
    # first request status update
    scheduler.update_from_output(scheduler_output0, model_runner_output)

    # Model output of the second request.
    model_runner_output = _make_model_runner_output(scheduler_output1)
    # second request status update
    scheduler.update_from_output(scheduler_output1, model_runner_output)

    # Schedule the next step again. The first request and second request
    # can be scheduled with decode phase.
    scheduler_output2 = scheduler.schedule()
    scheduled_cached_reqs = scheduler_output2.scheduled_cached_reqs
    assert scheduled_cached_reqs[0].new_block_ids[0] == exp_cached0_new
    assert scheduled_cached_reqs[1].new_block_ids[0] == exp_cached1_new

