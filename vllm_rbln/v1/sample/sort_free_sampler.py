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
from vllm_rbln.logger import init_logger
from vllm.v1.sample.metadata import SamplingMetadata
from vllm.v1.sample.sampler import Sampler as VLLMSampler
import rebel
from vllm_rbln.v1.sample.ops.penalties import (apply_all_penalties as
                                               rbln_apply_all_penalties)

logger = init_logger(__name__)

_SAMPLING_EPS = 1e-5


def dual_pivot_top_p_sample(
    probs: torch.Tensor,
    top_p: torch.Tensor,
    max_rounds: int = 64,
) -> torch.Tensor:
    """
    Simple torch implementation of `Dual Pivot Rejection Sampling`
    for top-p (nucleus) sampling.
    It excludes complicated logic like thread block-level handling
    (e.g. block scan/reduce algorithm),
    shared memory access, etc.

    Args:
    - probs: (B, V) unnormalized probabilities (non-negative)
    - top_p: (B,) or float, top_p value per batch
    - max_rounds: maximum number of rounds to try (for safety)
    Returns: sampled indices of shape (B,)
    Notes:
    - We do `inverse transform sampling` over entries whose prob > low.
    - Then use two pivots (pivot0, pivot1) to shrink [low, high] until accepted.
    - Acceptance rule: sum_{i: p_i > pivot0} p_i < top_p  -> accept sampled_id
    """
    assert probs.dim() == 2, "probs must be (B, V)"
    B, V = probs.shape
    device = probs.device
    dtype = probs.dtype

    if isinstance(top_p, (int, float)):
        top_p = torch.full((B, ), float(top_p), device=device, dtype=dtype)
    else:
        top_p = top_p.to(device=device, dtype=dtype).view(B)

    low = torch.zeros(B, device=device, dtype=dtype)
    high = torch.ones(B, device=device, dtype=dtype)
    q = probs.sum(dim=1)
    accepted = torch.zeros(B, device=device, dtype=torch.bool)
    out_idx = torch.zeros(B, device=device, dtype=torch.long)

    arange_V = torch.arange(V, device=device)

    for _ in range(max_rounds):
        if accepted.all():
            break

        # 1. Inverse-transform sampling only for non-accepted rows
        act = ~accepted
        # Uniform u in [0, q)
        u = torch.rand(act.sum(), device=device, generator=None,
                       dtype=dtype) * q[act]
        # Build per-row masks and cumsum
        P = probs[act]  # (A, V)
        low_rows = low[act].unsqueeze(1)  # (A, 1)
        mask_low = (low_rows < P)  # (A, V)
        # Cumulative Distribution Function of probs where probs > low
        cdf = (P * mask_low).cumsum(dim=1)  # (A, V)
        # Get first index for sampled token(j) index
        # where CDF(j-1) ≤ u < CDF(j)
        ge = cdf >= u.unsqueeze(1)  # (A, V)
        any_ge = ge.any(dim=1)
        first_idx = torch.zeros_like(any_ge, dtype=torch.long)
        if any_ge.any():
            first_idx[any_ge] = ge[any_ge].float().argmax(dim=1)
        # Get the last valid index where mask_low is True
        # for rows where no CDF(j) ≥ u (Fallback)
        last_valid = torch.where(mask_low, arange_V,
                                 -1).max(dim=1).values.clamp_min(0)
        # Select sampled token index over active rows
        sampled_idx_active = torch.where(any_ge, first_idx, last_valid)  # (A,)
        # Write to out_idx
        out_idx[act] = sampled_idx_active
        sampled_p = P.gather(1, sampled_idx_active.unsqueeze(1)).squeeze(1)
        pivot0 = torch.zeros(B, device=device, dtype=dtype)
        pivot0[act] = sampled_p
        pivot1 = (pivot0 + high) * 0.5

        # 2. Compute mass_gt_pivot0, mass_gt_pivot1 per row
        mass_gt_pivot0 = ((probs > pivot0.unsqueeze(1)) * probs).sum(dim=1)
        mass_gt_pivot1 = ((probs > pivot1.unsqueeze(1)) * probs).sum(dim=1)

        # 3. Binary search to update [low, high] per row
        # Case 1: pivot0 accepted (use sampled id)
        case1 = (mass_gt_pivot0 < top_p) & (~accepted)
        accepted |= case1

        # Case 2: pivot0 rejected, pivot1 accepted
        case2 = (~accepted) & (mass_gt_pivot0 >= top_p) & (mass_gt_pivot1
                                                           < top_p)
        low = torch.where(case2, pivot0, low)
        high = torch.where(case2, pivot1, high)
        q = torch.where(case2, mass_gt_pivot0, q)

        # Case 3: both rejected
        case3 = (~accepted) & (mass_gt_pivot1 >= top_p)
        low = torch.where(case3, pivot1, low)
        q = torch.where(case3, mass_gt_pivot1, q)

    return out_idx


@torch.library.custom_op("rbln::top_p_only", mutates_args=())
def top_p_only(
    probs: torch.Tensor,
    top_p: torch.Tensor,
    max_rounds: int = 64,
) -> torch.Tensor:
    return dual_pivot_top_p_sample(probs, top_p, max_rounds)


@top_p_only.register_fake
def top_p_only_fake(
    probs: torch.Tensor,
    top_p: torch.Tensor,
    max_rounds: int = 64,
) -> torch.Tensor:
    return dual_pivot_top_p_sample(probs, top_p, max_rounds)


class Sampler(VLLMSampler):

    def __init__(self, seed):
        super().__init__()
        rebel.manual_seed(seed)

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
            random_sampled = torch.ops.rbln.top_p_only(probs,
                                                       sampling_metadata.top_p)

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
