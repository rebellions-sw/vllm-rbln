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
    requests = create_requests(num_requests=8,
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

if __name__=="__main__":
    test_add_requests()
    test_finish_request()
    # test_schedule_single_seq(None, None)
    test_schedule_multi_seq(None, None)