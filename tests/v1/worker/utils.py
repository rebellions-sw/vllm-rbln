# Copyright 2025 Rebellions Inc. All rights reserved.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at:

#     http://www.apache.org/licenses/LICENSE-2.0

from collections.abc import Callable
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from types import SimpleNamespace
from typing import Optional

import torch
import torch.nn as nn
from vllm.config import CacheConfig, ModelConfig, SchedulerConfig, VllmConfig
from vllm.multimodal.inputs import (MultiModalFeatureSpec,
                                    MultiModalKwargsItem, PlaceholderRange)
from vllm.platforms import current_platform
from vllm.sampling_params import SamplingParams
from vllm.v1.core.kv_cache_manager import KVCacheManager, Request
from vllm.v1.core.kv_cache_utils import get_request_block_hasher
from vllm.v1.core.sched.output import CachedRequestData, NewRequestData
from vllm.v1.kv_cache_interface import (FullAttentionSpec, KVCacheConfig,
                                        KVCacheGroupSpec)
from vllm.v1.request import RequestStatus
from vllm_rbln.v1.core.optimum_scheduler import RBLNSchedulerOutput
from vllm_rbln.v1.worker.optimum_input_batch import RBLNInputBatch
from vllm_rbln.v1.worker.optimum_model_runner import RBLNOptimumModelRunner

MAX_NUM_SEQ = 2
MAX_MODEL_LEN = 64
OB_SIZE = 16
IB_SIZE = 4
NUM_BLOCKS = MAX_MODEL_LEN // OB_SIZE * MAX_NUM_SEQ + 1
DEVICE = current_platform.device_type


def make_kv_cache_config(block_size: int, num_blocks: int) -> KVCacheConfig:
    return KVCacheConfig(
        num_blocks=num_blocks,
        kv_cache_tensors=[],
        kv_cache_groups=[
            KVCacheGroupSpec(
                ["layer"],
                FullAttentionSpec(block_size, 1, 1, torch.float32, False),
            )
        ],
    )


def make_request(
    request_id: str,
    prompt_token_ids: list[int],
    block_size: int,
    hash_fn: Callable,
    mm_positions: Optional[list[PlaceholderRange]] = None,
    mm_hashes: Optional[list[str]] = None,
    prompt_logprobs: Optional[int] = None,
    cache_salt: Optional[str] = None,
):
    mm_features = []
    if mm_positions is not None:
        for j, position in enumerate(mm_positions):
            identifier = mm_hashes[j] if mm_hashes else f"hash_{j}"
            mm_feature = MultiModalFeatureSpec(
                data=MultiModalKwargsItem.dummy("dummy_m"),
                mm_position=position,
                identifier=identifier,
                modality="image")
            mm_features.append(mm_feature)

    return Request(request_id=request_id,
                   prompt_token_ids=prompt_token_ids,
                   mm_features=mm_features if mm_features else None,
                   sampling_params=SamplingParams(
                       max_tokens=17, prompt_logprobs=prompt_logprobs),
                   pooling_params=None,
                   eos_token_id=100,
                   lora_request=None,
                   cache_salt=cache_salt,
                   block_hasher=get_request_block_hasher(block_size, hash_fn))


def finish_request(manager: KVCacheManager, request: Request):
    request.status = RequestStatus.FINISHED_ABORTED
    manager.free(request)


def initialize_kv_cache(runner: RBLNOptimumModelRunner):
    """
    Only perform necessary steps in RBLNOptimumModelRunner.initialize_kv_cache()
    """
    kv_cache_config = make_kv_cache_config(
        block_size=IB_SIZE,
        num_blocks=NUM_BLOCKS * (OB_SIZE // IB_SIZE),
    )
    runner.kv_cache_config = kv_cache_config
    runner.input_batch = RBLNInputBatch(
        max_num_reqs=runner.max_num_reqs,
        max_model_len=runner.max_model_len,
        max_num_batched_tokens=runner.max_num_tokens,
        device=runner.device,
        pin_memory=runner.pin_memory,
        vocab_size=runner.model_config.get_vocab_size(),
        block_sizes=[
            kv_cache_config.kv_cache_groups[0].kv_cache_spec.block_size
        ],
        is_spec_decode=False,
        logitsprocs=[],
        is_pooling_model=False,
        bucket_sizes=None,  # No RBLN sampler in tests
    )


def get_vllm_config(async_scheduling=False):
    scheduler_config = SchedulerConfig(
        max_num_seqs=MAX_NUM_SEQ,
        max_num_batched_tokens=MAX_MODEL_LEN,
        max_model_len=MAX_MODEL_LEN,
        async_scheduling=async_scheduling,
    )
    model_config = ModelConfig(
        model="facebook/opt-125m",
        dtype=torch.float,
        seed=42,
    )
    cache_config = CacheConfig(
        block_size=IB_SIZE,
        swap_space=0,
        cache_dtype="auto",
        enable_prefix_caching=True,
    )
    additional_config = {
        "attn_block_size": OB_SIZE,
    }
    vllm_config = VllmConfig(
        cache_config=cache_config,
        model_config=model_config,
        scheduler_config=scheduler_config,
        additional_config=additional_config,
    )
    return vllm_config


class MockModelWrapper(nn.Module):

    class MockModel:

        def __init__(self):
            self.kv_block_adapter = SimpleNamespace(
                get_available_num_blocks=lambda: NUM_BLOCKS)

    def __init__(self):
        super().__init__()
        self.model = self.MockModel()


def _schedule_new_request(
    *req_ids: str,
    token_ids: list[int],
    block_ids: tuple[list[int], ...],
    new_computed_tokens: int,
    finished_req_ids: Optional[list[str]] = None,
    new_computed_blocks: Optional[list[int]] = None,
    preempted_req_ids: Optional[list[str]] = None,
) -> RBLNSchedulerOutput:
    new_reqs = []
    num_scheduled_tokens = {}
    total_num_scheduled_tokens = 0
    for req_id in req_ids:
        new_reqs.append(
            NewRequestData(
                req_id=req_id,
                prompt_token_ids=token_ids,
                mm_kwargs=[],
                mm_hashes=[],
                mm_positions=[],
                sampling_params=SamplingParams(),
                pooling_params=None,
                block_ids=block_ids,
                num_computed_tokens=new_computed_tokens,
                lora_request=None,
            ))
        num_scheduled_tokens[req_id] = len(token_ids)
        total_num_scheduled_tokens += num_scheduled_tokens[req_id]

    return RBLNSchedulerOutput(
        scheduled_new_reqs=new_reqs,
        scheduled_cached_reqs=CachedRequestData.make_empty(),
        num_scheduled_tokens=num_scheduled_tokens,
        total_num_scheduled_tokens=total_num_scheduled_tokens,
        scheduled_spec_decode_tokens={},
        scheduled_encoder_inputs={},
        num_common_prefix_blocks=0,
        finished_req_ids=set(finished_req_ids) if finished_req_ids else set(),
        free_encoder_mm_hashes=[],
        structured_output_request_ids={},
        grammar_bitmask=None,
        new_computed_blocks=new_computed_blocks if new_computed_blocks else [],
        preempted_req_ids=preempted_req_ids if preempted_req_ids else [],
    )


def _schedule_cached_reqs(
    reqs: list[Request],
    new_block_ids: list[tuple[list[int], ...]],
    finished_req_ids: Optional[list[str]] = None,
    resumed_from_preemption: bool = False,
) -> RBLNSchedulerOutput:
    req_ids = []
    resumed_from_preemption = []
    arr_new_token_ids = []
    arr_num_computed_tokens = []
    num_scheduled_tokens = {}
    total_num_scheduled_tokens = 0
    for req in reqs:
        num_computed_tokens = req.num_computed_tokens
        new_token_ids = req.all_token_ids[num_computed_tokens:]
        req_ids.append(req.request_id)
        resumed_from_preemption.append(False)
        arr_new_token_ids.append(new_token_ids)
        arr_num_computed_tokens.append(num_computed_tokens)
        num_scheduled_tokens[req.request_id] = len(new_token_ids)
        total_num_scheduled_tokens += num_scheduled_tokens[req.request_id]

    cached_req_data = CachedRequestData(
        req_ids=req_ids,
        resumed_from_preemption=resumed_from_preemption,
        new_token_ids=arr_new_token_ids,
        new_block_ids=new_block_ids,
        num_computed_tokens=arr_num_computed_tokens,
    )

    return RBLNSchedulerOutput(
        scheduled_new_reqs=[],
        scheduled_cached_reqs=cached_req_data,
        num_scheduled_tokens=num_scheduled_tokens,
        total_num_scheduled_tokens=total_num_scheduled_tokens,
        scheduled_spec_decode_tokens={},
        scheduled_encoder_inputs={},
        num_common_prefix_blocks=0,
        finished_req_ids=set(finished_req_ids) if finished_req_ids else set(),
        free_encoder_mm_hashes=[],
        structured_output_request_ids={},
        grammar_bitmask=None,
    )
