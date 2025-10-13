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
# isort: off
import torch
import rebel.core.random
from vllm_rbln.logger import init_logger
from vllm.v1.sample.metadata import SamplingMetadata
from vllm.v1.sample.ops.topk_topp_sampler import random_sample
from vllm.v1.sample.sampler import Sampler as VLLMSampler

from vllm_rbln.v1.sample.ops.penalties import (apply_all_penalties as
                                               rbln_apply_all_penalties)

logger = init_logger(__name__)


def _random_sample(
    probs: torch.Tensor,
    generators: dict[int, torch.Generator],
) -> torch.Tensor:
    """Randomly sample from the probabilities.

    We use this function instead of torch.multinomial because torch.multinomial
    causes CPU-GPU synchronization.
    """
    q = torch.empty_like(probs)
    # NOTE(woosuk): To batch-process the requests without their own seeds,
    # which is the common case, we first assume that every request does
    # not have its own seed. Then, we overwrite the values for the requests
    # that have their own seeds.
    if len(generators) != probs.shape[0]:
        q.exponential_()
    if generators:
        # TODO(woosuk): This can be slow because we handle each request
        # one by one. Optimize this.
        for i, generator in generators.items():
            q[i].exponential_(generator=generator)
    return probs.div_(q).argmax(dim=-1).view(-1)


@torch.library.custom_op("rbln::top_p_only", mutates_args=())
def top_p_only(logits: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
    logits_sort, logits_idx = logits.sort(dim=-1, descending=False)
    probs_sort = logits_sort.softmax(dim=-1)
    probs_sum = torch.cumsum(probs_sort, dim=-1, out=probs_sort)
    top_p_mask = probs_sum <= 1 - p.unsqueeze(dim=1)
    # at least one
    top_p_mask[:, -1] = False
    logits_sort.masked_fill_(top_p_mask, -float("inf"))
    # Re-sort the probabilities.
    logits = logits_sort.scatter(dim=-1, index=logits_idx, src=logits_sort)
    probs = logits.softmax(dim=-1, dtype=torch.float32)
    return _random_sample(probs, {})


@top_p_only.register_fake
def top_p_only_fake(logits: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
    logits_sort, logits_idx = logits.sort(dim=-1, descending=False)
    probs_sort = logits_sort.softmax(dim=-1)
    probs_sum = torch.cumsum(probs_sort, dim=-1, out=probs_sort)
    top_p_mask = probs_sum <= 1 - p.unsqueeze(dim=1)
    # at least one
    top_p_mask[:, -1] = False
    logits_sort.masked_fill_(top_p_mask, -float("inf"))
    # Re-sort the probabilities.
    logits = logits_sort.scatter(dim=-1, index=logits_idx, src=logits_sort)
    probs = logits.softmax(dim=-1, dtype=torch.float32)
    return _random_sample(probs, {})

class Sampler(VLLMSampler):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        rebel.manual_seed(seed)

    def forward(self, logits: torch.Tensor,
                sampling_metadata: SamplingMetadata) -> torch.Tensor:
        return super().forward(logits, sampling_metadata)

    def sample(self, logits: torch.Tensor,
               sampling_metadata: SamplingMetadata) -> torch.Tensor:
        assert not (sampling_metadata.all_greedy
                    and sampling_metadata.all_random)

        if sampling_metadata.all_greedy:
            return self.greedy_sample(logits)

        # Currently, RBLN only supports top_p sampling.
        # Covering other cases with RBLN is work in progress.
        if (sampling_metadata.top_p is not None
                and sampling_metadata.top_k is None):
            # Apply temperature scaling if needed
            if sampling_metadata.temperature is not None:
                logits = logits / sampling_metadata.temperature.unsqueeze(-1)
            
            # Use native RBLN top_p_only operation
            probs = torch.nn.functional.softmax(logits, dim=-1)
            indices = torch.ops.rbln.top_p_only(probs, sampling_metadata.top_p)
            return indices

        return super().sample(logits, sampling_metadata)

    @torch.compiler.disable
    def apply_penalties(
        self,
        logits: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> torch.Tensor:
        if not sampling_metadata.no_penalties:
            assert sampling_metadata.prompt_token_ids is not None
            logits = rbln_apply_all_penalties(
                logits,
                sampling_metadata.prompt_token_ids,
                sampling_metadata.presence_penalties,
                sampling_metadata.frequency_penalties,
                sampling_metadata.repetition_penalties,
                sampling_metadata.output_token_ids,
            )
        return logits