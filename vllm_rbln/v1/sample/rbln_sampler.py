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
from typing import Optional
from vllm_rbln.logger import init_logger
from vllm.v1.sample.metadata import SamplingMetadata
from vllm.v1.sample.sampler import Sampler as VLLMSampler
import rebel
from vllm.config import LogprobsMode
from vllm.v1.sample.ops import apply_top_k_top_p
from vllm_rbln.v1.sample.ops.penalties import (apply_all_penalties as
                                               rbln_apply_all_penalties)
import vllm_rbln.rbln_envs as envs

logger = init_logger(__name__)

_SAMPLING_EPS = 1e-5


@torch.library.custom_op("rbln::top_k_top_p", mutates_args=())
def top_k_top_p(logits: torch.Tensor, k: torch.Tensor,
                p: torch.Tensor) -> torch.Tensor:
    return apply_top_k_top_p(logits, k, p)


@top_k_top_p.register_fake
def top_k_top_p_fake(logits: torch.Tensor, k: torch.Tensor,
                     p: torch.Tensor) -> torch.Tensor:
    return apply_top_k_top_p(logits, k, p)


class RBLNSampler(VLLMSampler):

    def __init__(self,
                 logprobs_mode: LogprobsMode = "raw_logprobs",
                 seed: int = 42):
        super().__init__()  # FIXME topk_topp_sampler duplicated?
        rebel.manual_seed(seed)

        options = {"compile_context": rebel.CompileContext()}
        if envs.VLLM_RBLN_COMPILE_STRICT_MODE:
            options["mode"] = "strict"
        self._compiled_rbln_topk_topp_sampler = torch.compile(
            self._rbln_topk_topp_sampler_impl,
            dynamic=False,
            fullgraph=True,
            backend="rbln",
            options=options,
        )
        self.logprobs_mode = logprobs_mode

    def apply_temperature(
        self,
        logits: torch.Tensor,
        temp: torch.Tensor,
    ) -> torch.Tensor:
        # NOTE:
        # in-place division triggers buffer key error
        # in torchinductor
        return logits.div(temp.unsqueeze(dim=1))

    def apply_topk_topp_sampler(
            self, logits: torch.Tensor, top_k: torch.Tensor,
            top_p: torch.Tensor
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        sampled = self.rbln_topk_topp_sampler(logits, top_k, top_p)
        logits_to_return = None
        if self.logprobs_mode == LogprobsMode.PROCESSED_LOGITS:
            logits_to_return = logits
        elif self.logprobs_mode == LogprobsMode.PROCESSED_LOGPROBS:
            logits_to_return = logits.log_softmax(dim=-1, dtype=torch.float32)
        return sampled, logits_to_return

    @staticmethod
    def _rbln_topp_sampler_impl(logits: torch.Tensor,
                                top_p: torch.Tensor) -> torch.Tensor:
        """
        Implementation of RBLN top-p sampling.
        To avoid self parameter issues when torch.compile is used,
        we define this as a static method.
        """
        # Apply top-p sampling using RBLN custom op.
        # It requires softmax prior to calling the op.
        probs = torch.nn.functional.softmax(logits, dim=-1)
        sampled = torch.ops.rbln.top_p_only(probs, top_p)
        return sampled

    @staticmethod
    def _rbln_topk_topp_sampler_impl(logits: torch.Tensor, top_k: torch.Tensor,
                                     top_p: torch.Tensor) -> torch.Tensor:
        """
        Implementation of RBLN top-k top-p sampling.
        To avoid self parameter issues when torch.compile is used,
        we define this as a static method.
        """
        # Apply top-k top-p sampling using RBLN custom op.
        # It requires softmax prior to calling the op.
        probs = torch.nn.functional.softmax(logits, dim=-1)
        sampled = torch.ops.rbln.top_k_top_p(probs, top_k, top_p)
        return sampled

    @torch.compiler.disable
    def rbln_topk_topp_sampler(self, logits: torch.Tensor, top_k: torch.Tensor,
                               top_p: torch.Tensor) -> torch.Tensor:
        """
        Wrapper for the compiled RBLN top-p sampler.
        To avoid recompile on runtime, we decorate this method with
        `torch.compiler.disable` and call the pre-compiled function.
        """
        return self._compiled_rbln_topk_topp_sampler(logits, top_k, top_p)

    def sample(
        self,
        logits: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
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
                processed_logprobs = None
                if sampling_metadata.max_num_logprobs is not None:
                    if self.logprobs_mode == LogprobsMode.PROCESSED_LOGITS:
                        processed_logprobs = logits
                    elif self.logprobs_mode == LogprobsMode.PROCESSED_LOGPROBS:
                        processed_logprobs = self.compute_logprobs(logits)
                return greedy_sampled, processed_logprobs

        assert sampling_metadata.temperature is not None

        # Apply temperature.
        logits = self.apply_temperature(logits, sampling_metadata.temperature)

        # Apply logits processors that only apply to random sampling
        # (argmax invariant)
        for processor in sampling_metadata.logitsprocs.argmax_invariant:
            logits = processor.apply(logits)

        random_sampled, processed_logprobs = self.apply_topk_topp_sampler(
            logits, sampling_metadata.top_k, sampling_metadata.top_p)

        if greedy_sampled is None:
            return random_sampled, processed_logprobs

        sampled = torch.where(
            sampling_metadata.temperature < _SAMPLING_EPS,
            greedy_sampled,
            random_sampled,
            out=greedy_sampled,  # Reuse tensor
        )
        return sampled, processed_logprobs

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

    @staticmethod
    def get_bucket_sizes(max_num_seqs: int) -> list[int]:
        """Get the bucket sizes for the sampler.
        Args:
            max_num_seqs (int): The maximum number of sequences.
        Returns:
            list[int]: The bucket sizes.
        [1, 2, 4] + list(range(8, 256, 8)) + list(
            range(256, max_num_seqs + 1, 16))
        """
        # FIXME(eunji.lee)
        # Not used. To be removed.
        bucket_sizes = [i for i in [1, 2, 4] if i <= max_num_seqs]
        if max_num_seqs >= 8:
            # Step size 8 for small batch sizes, up to 256(not included)
            bucket_sizes += list(range(8, min(max_num_seqs + 1, 256), 8))
        if max_num_seqs >= 256:
            # Step size 16 for larger batch sizes
            bucket_sizes += list(range(256, max_num_seqs + 1, 16))
        return bucket_sizes


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
