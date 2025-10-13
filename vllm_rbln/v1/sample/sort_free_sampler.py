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


# It is fake implementation for torch.compile
# For implementation, refer to the flash-infer repository.
@torch.library.custom_op("rbln::top_p_only", mutates_args=())
def top_p_only(probs: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
    return _random_sample(probs, {})


@top_p_only.register_fake
def top_p_only_fake(probs: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
    return _random_sample(probs, {})


class Sampler(VLLMSampler):

    def __init__(self, seed):
        super().__init__()
        rebel.core.random.manual_seed(seed)

    def forward(self, logits: torch.Tensor,
                sampling_metadata: SamplingMetadata) -> torch.Tensor:
        return super().forward(logits, sampling_metadata)

    def sample(self, logits: torch.Tensor,
               sampling_metadata: SamplingMetadata) -> torch.Tensor:
        """Sample logits based on sampling metadata.

        The various logits processing functions called in this method
        may update the logits tensor in-place.
        """

        assert not (sampling_metadata.all_greedy
                    and sampling_metadata.all_random)
        if sampling_metadata.all_random:
            greedy_sampled = None
        else:
            greedy_sampled = self.greedy_sample(logits)
            if sampling_metadata.all_greedy:
                return greedy_sampled

        assert sampling_metadata.temperature is not None

        # Apply temperature.
        logits = self.apply_temperature(logits, sampling_metadata.temperature)

        # Apply min_p.
        if sampling_metadata.min_p is not None:
            logits = self.apply_min_p(logits, sampling_metadata.min_p)

        # Currently, RBLN only supports top_p sampling.
        # Covering other cases with RBLN is work in progress.
        if (sampling_metadata.top_p is not None
                and sampling_metadata.top_k is None):
            # Apply temperature scaling if needed
            if sampling_metadata.temperature is not None:
                logits = logits / sampling_metadata.temperature.unsqueeze(-1)

            # Use native RBLN top_p_only operation
            probs = torch.nn.functional.softmax(logits, dim=-1)
            random_sampled = torch.ops.rbln.top_p_only(probs, sampling_metadata.top_p)

        elif sampling_metadata.top_k is not None:
            # Apply top_k and/or top_p.
            random_sampled = self.topk_topp_sampler(
                logits,
                sampling_metadata.generators,
                sampling_metadata.top_k,
                sampling_metadata.top_p,
            )

        if greedy_sampled is None:
            return random_sampled

        sampled = torch.where(
            sampling_metadata.temperature < _SAMPLING_EPS,
            greedy_sampled,
            random_sampled,
            out=greedy_sampled,  # Reuse tensor
        )

        return sampled

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
