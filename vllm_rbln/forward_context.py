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
from typing import Any

import torch
import torch.distributed as dist
import vllm.forward_context as vfc
from vllm.config import CUDAGraphMode, VllmConfig
from vllm.forward_context import (BatchDescriptor, DPMetadata,
                                  batchsize_logging_interval,
                                  create_forward_context,
                                  override_forward_context, track_batchsize)
from vllm.v1.worker.ubatch_utils import UBatchSlices

import vllm_rbln.rbln_envs as envs
from vllm_rbln.logger import init_logger

logger = init_logger(__name__)


@dataclass
class RBLNDPMetadata(DPMetadata):
    max_pads_across_dp: torch.Tensor | None = None

    @staticmethod
    def num_tokens_across_dp(num_tokens: int, dp_size: int,
                             dp_rank: int) -> torch.Tensor:
        """
        Gather the num_tokens across all DP ranks and return results in a
        CPU tensor of size dp_size.
        """
        num_tokens_across_dp = [0] * dp_size
        num_tokens_across_dp[dp_rank] = num_tokens
        num_tokens_tensor = torch.tensor(num_tokens_across_dp,
                                         device="cpu",
                                         dtype=torch.int32)
        from vllm.distributed.parallel_state import get_dp_group
        dist.all_reduce(num_tokens_tensor, group=get_dp_group().cpu_group)
        return num_tokens_tensor

    @staticmethod
    def make(
        vllm_config: VllmConfig,
        num_tokens: int,
        num_padded_tokens: int | None = None,
    ) -> "RBLNDPMetadata":
        parallel_config = vllm_config.parallel_config
        dp_size = parallel_config.data_parallel_size
        dp_rank = parallel_config.data_parallel_rank

        if dp_size > 1:
            scheduler_config = vllm_config.scheduler_config
            max_pad = scheduler_config.max_num_batched_tokens
            batchsize = num_tokens

            if envs.VLLM_RBLN_SPECIALIZE_MOE_DECODE:
                num_padded_tokens = num_padded_tokens or max_pad

                pad_flag = 1 << 16
                pad_mask = pad_flag - 1
                assert max_pad % num_padded_tokens == 0 and pad_flag > max_pad

                if max_pad == num_padded_tokens:
                    batchsize |= pad_flag

                tokens_across_dp_cpu = RBLNDPMetadata.num_tokens_across_dp(
                    batchsize, dp_size, dp_rank)
                max_across_dp = torch.max(tokens_across_dp_cpu)
                max_pad = max_pad if max_across_dp > pad_flag \
                    else num_padded_tokens

                mask_tensor = torch.tensor([pad_mask] * dp_size,
                                           device="cpu",
                                           dtype=torch.int32)
                num_tokens_across_dp_cpu = tokens_across_dp_cpu & mask_tensor
            else:
                num_tokens_across_dp_cpu = RBLNDPMetadata.num_tokens_across_dp(
                    batchsize, dp_size, dp_rank)

            max_tokens_across_dp_cpu = torch.max(num_tokens_across_dp_cpu)
            max_pads_across_dp = torch.empty(max_pad, device="cpu")
        else:
            num_tokens_across_dp_cpu = torch.tensor([num_tokens],
                                                    device="cpu",
                                                    dtype=torch.int32)
            max_tokens_across_dp_cpu = num_tokens
            max_pads_across_dp = None

        return RBLNDPMetadata(max_tokens_across_dp_cpu,
                              num_tokens_across_dp_cpu,
                              max_pads_across_dp=max_pads_across_dp)


@contextmanager
def _set_forward_context(
    attn_metadata: Any,
    vllm_config: VllmConfig,
    virtual_engine: int = 0,
    num_tokens: int | None = None,
    num_tokens_across_dp: torch.Tensor | None = None,
    cudagraph_runtime_mode: CUDAGraphMode = CUDAGraphMode.NONE,
    batch_descriptor: BatchDescriptor | None = None,
    ubatch_slices: UBatchSlices | None = None,
    num_padded_tokens: int | None = None,
):
    """A context manager that stores the current forward context,
    can be attention metadata, etc.
    Here we can inject common logic for every model forward pass.
    """
    need_to_track_batchsize = track_batchsize and attn_metadata is not None
    if need_to_track_batchsize:
        vfc.forward_start_time = time.perf_counter()

    dp_metadata: DPMetadata | None = None
    enable_dp = vllm_config.parallel_config.data_parallel_size > 1
    use_moe_tokens_mask = envs.VLLM_RBLN_USE_MOE_TOKENS_MASK
    if (enable_dp or use_moe_tokens_mask) and (attn_metadata is not None
                                               or num_tokens is not None):
        dp_metadata = RBLNDPMetadata.make(vllm_config, num_tokens or 0,
                                          num_padded_tokens)

    forward_context = create_forward_context(
        attn_metadata,
        vllm_config,
        virtual_engine,
        dp_metadata,
        cudagraph_runtime_mode,
        batch_descriptor,
        ubatch_slices,
    )

    try:
        with override_forward_context(forward_context):
            yield
    finally:
        if need_to_track_batchsize:
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


vfc.set_forward_context = _set_forward_context
