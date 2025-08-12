# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from typing import Optional
from unittest.mock import Mock

import pytest
# import torch

from vllm.config import (CacheConfig, KVTransferConfig, ModelConfig,
                         SchedulerConfig, SpeculativeConfig, VllmConfig)
from vllm.multimodal.inputs import MultiModalKwargs, PlaceholderRange
from vllm.sampling_params import GuidedDecodingParams, SamplingParams
from vllm.v1.core.sched.output import CachedRequestData, SchedulerOutput
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.kv_cache_interface import (FullAttentionSpec, KVCacheConfig,
                                        KVCacheGroupSpec)
from vllm.v1.outputs import ModelRunnerOutput
from vllm.v1.request import Request, RequestStatus
from vllm.v1.structured_output import StructuredOutputManager
from vllm.v1.structured_output.request import StructuredOutputRequest

# from .utils import create_requests, create_scheduler
from utils import create_requests, create_scheduler


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

@pytest.mark.parametrize("enable_prefix_caching, prompt_logprobs", [
    (None, None),
])
def test_schedule_single_seq(enable_prefix_caching: Optional[bool],
                  prompt_logprobs: Optional[int]):
    '''Test scheduling.
    Only one case: default APC/no prompt logprobs
    '''
    scheduler = create_scheduler(enable_prefix_caching=enable_prefix_caching, max_num_seqs=1)
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
    assert len(scheduler.waiting) == len(requests)-1
    assert len(scheduler.running) == 1

    for i, running_request in enumerate(scheduler.running):
        assert requests[i] == running_request
        scheduler.finish_requests(running_request.request_id,
                                  RequestStatus.FINISHED_ABORTED)
        assert running_request.request_id not in scheduler.requests, print(f"{running_request.request_id} must be not in {scheduler.requests}")

@pytest.mark.parametrize("enable_prefix_caching, prompt_logprobs", [
    (None, None),
])        
def test_schedule_multi_seq(enable_prefix_caching: Optional[bool],
                            prompt_logprobs: Optional[int]):
    '''Test scheduling.
    Only one case: default APC/no prompt logprobs
    '''
    scheduler = create_scheduler(enable_prefix_caching=enable_prefix_caching, max_num_seqs=2)
    requests = create_requests(num_requests=2,
                               prompt_logprobs=prompt_logprobs)

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
    assert len(scheduler.waiting) == len(requests) - scheduler.max_num_running_reqs
    assert len(scheduler.running) == scheduler.max_num_running_reqs
    
    batch = []
    for i, running_request in enumerate(scheduler.running):
        batch.append(running_request.request_id)
        assert requests[i] == running_request

    scheduler.finish_requests(batch,
                              RequestStatus.FINISHED_ABORTED)
    for request_id in batch:
        assert request_id not in scheduler.requests, print(f"{request_id} must be not in {scheduler.requests}")

@pytest.mark.parametrize("enable_prefix_caching, prompt_logprobs", [
    (None, None),
])
def test_schedule_concurrent_batches(enable_prefix_caching: Optional[bool],
                                     prompt_logprobs: Optional[int]):
    scheduler = create_scheduler(
        max_num_batched_tokens=128,
        max_num_seqs=2,                              # 이게 vllm-rbln에서 가능한 시나리오?
        enable_prefix_caching=enable_prefix_caching,
    )
    requests = create_requests(
        num_requests=4,
        num_tokens=64,
        prompt_logprobs=prompt_logprobs,
    )

    # Schedule the first request.
    scheduler.add_request(requests[0])
    scheduler_output0 = scheduler.schedule()
    assert len(scheduler_output0.scheduled_new_reqs) == 1
    assert scheduler_output0.num_scheduled_tokens[
        requests[0].request_id] == 64

    import pdb; pdb.set_trace()
    
    # The first request is still running, so only schedule the second request.
    scheduler.add_request(requests[1])
    scheduler_output1 = scheduler.schedule()
    assert len(scheduler_output1.scheduled_new_reqs) == 1
    assert scheduler_output1.num_scheduled_tokens[
        requests[1].request_id] == 64

    # Model output of the first request.
    model_runner_output = ModelRunnerOutput(
        req_ids=[requests[0].request_id],
        req_id_to_index={requests[0].request_id: 0},
        sampled_token_ids=[[0]],
        spec_token_ids=None,
        logprobs=None,
        prompt_logprobs_dict={},
    )
    
    import pdb; pdb.set_trace()
    scheduler.update_from_output(scheduler_output0, model_runner_output)

    import pdb; pdb.set_trace()
    # Schedule the next step.
    # The first request can be scheduled again while the second
    # request is still running.
    scheduler_output2 = scheduler.schedule() # NOTE(si) 아 이게 decoder phase 인가?
    assert scheduler_output2.num_scheduled_tokens[requests[0].request_id] == 1

    import pdb; pdb.set_trace()
    # Model output of the second request.
    model_runner_output = ModelRunnerOutput(
        req_ids=[requests[1].request_id],
        req_id_to_index={requests[1].request_id: 0},
        sampled_token_ids=[[0]],
        spec_token_ids=None,
        logprobs=None,
        prompt_logprobs_dict={},
    ) # 모델의 결과
    # 결과를 가지고 
    # 1. 생성된 토큰 추가 2. EOS 인 경우 FINISHED로 처리 및 block 해제 3. preemption 및 swap 수행
    scheduler.update_from_output(scheduler_output1, model_runner_output) 
    
    import pdb; pdb.set_trace()


# def test_preempt_during_execution():
#     # NOTE(woosuk): The actual number of available blocks is 10 instead of 11
#     # because block 0 is reserved as the null block.
#     scheduler = create_scheduler(max_num_batched_tokens=100,
#                                  block_size=16,
#                                  num_blocks=11,
#                                  enable_prefix_caching=False)
#     requests = create_requests(num_requests=2, num_tokens=80)

#     # Schedule the first request.
#     scheduler.add_request(requests[0])
#     scheduler_output0 = scheduler.schedule()
#     assert len(scheduler_output0.num_scheduled_tokens) == 1
#     assert len(scheduler_output0.scheduled_new_reqs[0].block_ids[0]) == 5

#     # Schedule the second request while the first request is still running.
#     # This scenario can occur in certain cases, when max_concurrent_batches > 1
#     # (e.g., when pipeline parallelism is used).
#     scheduler.add_request(requests[1])
#     scheduler_output1 = scheduler.schedule()
#     assert len(scheduler_output1.num_scheduled_tokens) == 1
#     assert len(scheduler_output1.scheduled_new_reqs[0].block_ids[0]) == 5

#     # Get the output of the first request.
#     model_runner_output0 = ModelRunnerOutput(
#         req_ids=[requests[0].request_id],
#         req_id_to_index={requests[0].request_id: 0},
#         sampled_token_ids=[[0]],
#         spec_token_ids=None,
#         logprobs=None,
#         prompt_logprobs_dict={},
#     )
#     scheduler.update_from_output(scheduler_output0, model_runner_output0)

#     # Schedule the first request again. This will cause the preemption
#     # of the second request because the KV cache is full.
#     _ = scheduler.schedule()
#     assert len(scheduler.running) == 1
#     assert scheduler.running[0] == requests[0]
#     assert requests[1].status == RequestStatus.PREEMPTED

#     model_runner_output1 = ModelRunnerOutput(
#         req_ids=[requests[1].request_id],
#         req_id_to_index={requests[1].request_id: 0},
#         sampled_token_ids=[[42]],
#         spec_token_ids=None,
#         logprobs=None,
#         prompt_logprobs_dict={},
#     )
#     scheduler.update_from_output(scheduler_output1, model_runner_output1)

#     # The second request (that is preempted) should be updated with the
#     # sampled token id.
#     assert len(requests[1].output_token_ids) == 1
#     assert requests[1].output_token_ids[0] == 42

# TODO(si) remove
if __name__=="__main__":
    # test_add_requests()
    # test_finish_request()
    # test_schedule_single_seq(None, None)
    # test_schedule_multi_seq(None, None)
    test_schedule_concurrent_batches(None, None)