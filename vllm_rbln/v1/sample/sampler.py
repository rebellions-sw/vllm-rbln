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
from vllm.logger import init_logger
from vllm.v1.sample.metadata import SamplingMetadata
from vllm.v1.sample.ops.topk_topp_sampler import random_sample
from vllm.v1.sample.sampler import Sampler as VLLMSampler

from vllm_rbln.v1.sample.ops.penalties import (apply_all_penalties as
                                               rbln_apply_all_penalties)

logger = init_logger(__name__)


class Sampler(VLLMSampler):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._rbln_top_p_sample = torch.compile(
            self._rbln_top_p_sample_only,
            dynamic=False,
            fullgraph=False,
            backend="rbln",
        )

    @torch.compiler.disable
    def rbln_top_p_sample(self, sorted_logits: torch.Tensor,
                          sampling_metadata: SamplingMetadata) -> torch.Tensor:
        return self._rbln_top_p_sample(sorted_logits, sampling_metadata)

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
            sorted_logits, sorted_indices = torch.sort(logits,
                                                       descending=False,
                                                       dim=-1)
            sorted_probs = self.rbln_top_p_sample(sorted_logits,
                                                  sampling_metadata)
            # scatter later to avoid graph breaking on NPU
            probs = sorted_probs.scatter(1, sorted_indices, sorted_probs)
            return random_sample(probs, sampling_metadata.generators)

        return super().sample(logits, sampling_metadata)

    def _rbln_top_p_sample_only(
            self, sorted_logits: torch.Tensor,
            sampling_metadata: SamplingMetadata) -> torch.Tensor:
        if sampling_metadata.temperature is not None:
            sorted_logits = sorted_logits.div_(
                sampling_metadata.temperature.unsqueeze(-1))

        # Convert to probabilities
        sorted_probs = torch.softmax(sorted_logits, dim=-1)
        # Calculate cumulative probabilities
        cumsum_probs = torch.cumsum(sorted_probs, dim=-1)
        # Create mask for tokens to keep (cumulative probability <= top_p)
        top_p_threshold = sampling_metadata.top_p.unsqueeze(-1)
        mask = cumsum_probs <= 1 - top_p_threshold

        # Always keep at least the first token
        mask[..., -1] = False

        sorted_logits.masked_fill_(mask, float("-inf"))
        sorted_probs = torch.softmax(sorted_logits, dim=-1)
        return sorted_probs

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


WARM_UP_CONFIGS = [
    {
        "name": "no_penalty_greedy",
        "no_penalties": True,
        "all_greedy": True,
        "all_random": False,
        "temperature": 0.0
    },
    {
        "name": "no_penalty_topp",
        "no_penalties": True,
        "all_greedy": False,
        "all_random": True,
        "top_p": 0.9,
        "temperature": 0.5
    },
    {
        "name": "no_penalty_topk",
        "no_penalties": True,
        "all_greedy": False,
        "all_random": True,
        "top_k": 1.0,
        "temperature": 0.5
    },
    {
        "name": "no_penalty_topp_topk",
        "no_penalties": True,
        "all_greedy": False,
        "all_random": True,
        "top_p": 0.9,
        "top_k": 1.0,
        "temperature": 0.5
    },
    {
        "name": "penalty_greedy",
        "no_penalties": False,
        "frequency_penalties": 0.1,
        "presence_penalties": 0.1,
        "repetition_penalties": 1.0,
        "all_greedy": True,
        "all_random": False,
        "temperature": 0.0
    },
    {
        "name": "penalty_topp",
        "no_penalties": False,
        "frequency_penalties": 0.1,
        "presence_penalties": 0.1,
        "repetition_penalties": 1.0,
        "all_greedy": False,
        "all_random": True,
        "top_p": 0.9,
        "temperature": 0.5
    },
    {
        "name": "penalty_topk",
        "no_penalties": False,
        "frequency_penalties": 0.1,
        "presence_penalties": 0.1,
        "repetition_penalties": 1.0,
        "all_greedy": False,
        "all_random": True,
        "top_k": 1.0,
        "temperature": 0.5
    },
    {
        "name": "penalty_topp_topk",
        "no_penalties": False,
        "frequency_penalties": 0.1,
        "presence_penalties": 0.1,
        "repetition_penalties": 1.0,
        "all_greedy": False,
        "all_random": True,
        "top_p": 0.9,
        "top_k": 1.0,
        "temperature": 0.5
    },
]
