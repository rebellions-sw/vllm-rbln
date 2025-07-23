import torch
from vllm.v1.sample.metadata import SamplingMetadata
from vllm.v1.sample.sampler import Sampler as VLLMSampler

from vllm_rbln.v1.sample.ops.penalties import (
    apply_all_penalties as rbln_apply_all_penalties,
)


@torch.compiler.disable
def multinomial(prob: torch.Tensor, num_samples: int = 1) -> torch.Tensor:
    # compile is disabled since multinomial is currently not supported on RBLN.
    sampled_ids = torch.multinomial(prob, num_samples=num_samples)
    sampled_ids = sampled_ids.view(prob.shape[:-1] + (1,))
    return sampled_ids


class Sampler(VLLMSampler):
    def forward(
        self, logits: torch.Tensor, sampling_metadata: SamplingMetadata
    ) -> torch.Tensor:
        return super().forward(logits, sampling_metadata)

    def sample(
        self, logits: torch.Tensor, sampling_metadata: SamplingMetadata
    ) -> torch.Tensor:
        assert not (
            sampling_metadata.all_greedy and sampling_metadata.all_random
        )

        if sampling_metadata.all_greedy:
            return self.greedy_sample(logits)

        # Currently, RBLN only supports top_p sampling.
        # Covering other cases with RBLN is work in progress.
        if (
            sampling_metadata.top_k is None
            and sampling_metadata.top_p is not None
        ):
            return self._rbln_top_p_sample_only(logits, sampling_metadata)

        return super().sample(logits, sampling_metadata)

    def _rbln_top_p_sample_only(
        self, logits: torch.Tensor, sampling_metadata: SamplingMetadata
    ) -> torch.Tensor:
        sorted_logits, sorted_indices = torch.sort(
            logits, descending=False, dim=-1
        )
        if sampling_metadata.temperature is not None:
            sorted_logits = sorted_logits.div_(sampling_metadata.temperature)

        # Convert to probabilities
        sorted_probs = torch.softmax(sorted_logits, dim=-1)
        # Calculate cumulative probabilities
        cumsum_probs = torch.cumsum(sorted_probs, dim=-1)
        # Create mask for tokens to keep (cumulative probability <= top_p)
        top_p_threshold = sampling_metadata.top_p
        mask = cumsum_probs <= top_p_threshold

        # Always keep at least the first token
        mask[..., -1] = False

        sorted_logits.masked_fill_(mask, float("-inf"))
        sorted_probs = torch.softmax(sorted_logits, dim=-1)

        # scatter later to avoid graph breaking on NPU
        probs = sorted_probs.scatter(1, sorted_indices, sorted_probs)

        return multinomial(probs.view(-1, probs.size(-1)), num_samples=1)

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
