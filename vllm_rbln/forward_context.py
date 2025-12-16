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

import time
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Optional

import torch
import vllm.forward_context as vfc
from vllm.config import CUDAGraphMode, VllmConfig
from vllm.forward_context import (BatchDescriptor, DPMetadata, ForwardContext,
                                  batchsize_logging_interval, track_batchsize)

import vllm_rbln.rbln_envs as envs
from vllm_rbln.logger import init_logger

logger = init_logger(__name__)


@dataclass
class RBLNDPMetadata(DPMetadata):
    max_pads_across_dp: int = 0
    num_tokens_across_dp_cpu: Optional[torch.Tensor] = None

    @staticmethod
    def make(
        vllm_config: VllmConfig,
        attn_metadata: Any,
        num_tokens: int,
        num_tokens_across_dp: Optional[torch.Tensor] = None
    ) -> "RBLNDPMetadata":

        parallel_config = vllm_config.parallel_config
        dp_size = parallel_config.data_parallel_size
        dp_rank = parallel_config.data_parallel_rank

        scheduler_config = vllm_config.scheduler_config
        max_pad = scheduler_config.max_num_batched_tokens

        if attn_metadata is not None and hasattr(attn_metadata,
                                                 "num_prefill_tokens"):
            # for v0 attention backends
            batchsize = attn_metadata.num_prefill_tokens + \
                attn_metadata.num_decode_tokens

            disable_dp = dp_size == 1
            use_dummy_prefill = envs.VLLM_RBLN_DP_IMPL == "dummy_prefill"
            if (disable_dp or use_dummy_prefill) and \
                attn_metadata.num_decode_tokens > 0:
                max_pad = scheduler_config.max_num_seqs
        else:
            # for v1 attention backends or no attn_metadata
            batchsize = num_tokens

        # If num_tokens_across_dp is None, it will be computed by all_reduce
        # Otherwise, num_tokens_across_dp[dp_rank] should be equal to batchsize
        assert (num_tokens_across_dp is None
                or num_tokens_across_dp[dp_rank] == batchsize)
        if num_tokens_across_dp is None:
            num_tokens_across_dp = DPMetadata.num_tokens_across_dp(
                batchsize, dp_size, dp_rank)
        max_tokens_across_dp_cpu = torch.max(num_tokens_across_dp)
        cu_tokens_across_dp_cpu = torch.cumsum(num_tokens_across_dp, dim=0)
        return RBLNDPMetadata(max_tokens_across_dp_cpu,
                              cu_tokens_across_dp_cpu,
                              max_pads_across_dp=max_pad,
                              num_tokens_across_dp_cpu=num_tokens_across_dp)


@contextmanager
def _set_forward_context(
        attn_metadata: Any,
        vllm_config: VllmConfig,
        virtual_engine: int = 0,
        num_tokens: Optional[int] = None,
        num_tokens_across_dp: Optional[torch.Tensor] = None,
        cudagraph_runtime_mode: CUDAGraphMode = CUDAGraphMode.NONE,
        batch_descriptor: Optional[BatchDescriptor] = None):
    """A context manager that stores the current forward context,
    can be attention metadata, etc.
    Here we can inject common logic for every model forward pass.
    """
    need_to_track_batchsize = track_batchsize and attn_metadata is not None
    if need_to_track_batchsize:
        vfc.forward_start_time = time.perf_counter()
    dp_metadata: Optional[DPMetadata] = None
    enable_dp = vllm_config.parallel_config.data_parallel_size > 1
    use_moe_tokens_mask = envs.VLLM_RBLN_USE_MOE_TOKENS_MASK
    if (enable_dp or use_moe_tokens_mask) and (attn_metadata is not None
                                               or num_tokens is not None):
        dp_metadata = RBLNDPMetadata.make(vllm_config, attn_metadata,
                                          num_tokens or 0,
                                          num_tokens_across_dp)

    prev_context = vfc._forward_context
    vfc._forward_context = ForwardContext(
        no_compile_layers=vllm_config.compilation_config.
        static_forward_context,
        virtual_engine=virtual_engine,
        attn_metadata=attn_metadata,
        dp_metadata=dp_metadata,
        cudagraph_runtime_mode=cudagraph_runtime_mode,
        batch_descriptor=batch_descriptor,
    )

    try:
        yield
    finally:
        if need_to_track_batchsize:
            if hasattr(attn_metadata, "num_prefill_tokens"):
                # for v0 attention backends
                batchsize = attn_metadata.num_prefill_tokens + \
                    attn_metadata.num_decode_tokens
            else:
                # for v1 attention backends
                batchsize = num_tokens
            # we use synchronous scheduling right now,
            # adding a sync point here should not affect
            # scheduling of the next batch
            from vllm.platforms import current_platform
            synchronize = current_platform.synchronize
            if synchronize is not None:
                synchronize()
            now = time.perf_counter()
            # time measurement is in milliseconds
            vfc.batchsize_forward_time[batchsize].append(
                (now - vfc.forward_start_time) * 1000)
            if now - vfc.last_logging_time > batchsize_logging_interval:
                vfc.last_logging_time = now
                forward_stats = []
                for bs, times in vfc.batchsize_forward_time.items():
                    if len(times) <= 1:
                        # can be cudagraph / profiling run
                        continue
                    medium = torch.quantile(torch.tensor(times), q=0.5).item()
                    medium = round(medium, 2)
                    forward_stats.append((bs, len(times), medium))
                forward_stats.sort(key=lambda x: x[1], reverse=True)
                if forward_stats:
                    logger.info(("Batchsize forward time stats "
                                 "(batchsize, count, median_time(ms)): %s"),
                                forward_stats)

        vfc._forward_context = prev_context


vfc.set_forward_context = _set_forward_context
