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

import torch

from vllm_rbln.v1.sample.rbln_sampler import apply_top_k_top_p


class TestApplyTopKTopP:
    def test_none_none_returns_clone(self):
        """When both k and p are None, should return a clone of logits."""
        logits = torch.randn(2, 10)
        result = apply_top_k_top_p(logits, None, None)
        assert torch.equal(result, logits)
        # Must be a clone, not the same tensor
        assert result.data_ptr() != logits.data_ptr()

    def test_top_k_only(self):
        """Top-k should mask all but the top k logits."""
        logits = torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0]])
        k = torch.tensor([2])
        result = apply_top_k_top_p(logits, k, None)
        # Only top 2 values (4.0, 5.0) should remain, rest -inf
        assert result[0, 4].item() == 5.0
        assert result[0, 3].item() == 4.0
        assert result[0, 0].item() == float("-inf")
        assert result[0, 1].item() == float("-inf")
        assert result[0, 2].item() == float("-inf")

    def test_top_k_equals_vocab(self):
        """When k equals vocab size, no masking should occur."""
        logits = torch.tensor([[1.0, 2.0, 3.0]])
        k = torch.tensor([3])
        result = apply_top_k_top_p(logits, k, None)
        # All logits should remain
        assert torch.allclose(result, logits)

    def test_top_p_only(self):
        """Top-p should mask low probability tokens."""
        # Create logits where softmax gives clear probabilities
        logits = torch.tensor([[10.0, 1.0, 0.0, -1.0, -10.0]])
        p = torch.tensor([0.9])
        result = apply_top_k_top_p(logits, None, p)
        # The highest logit should always survive
        assert result[0, 0].item() == 10.0
        # Very low probability tokens should be masked
        assert result[0, 4].item() == float("-inf")

    def test_top_p_one_keeps_all(self):
        """p=1.0 should keep all tokens (or nearly all)."""
        logits = torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0]])
        p = torch.tensor([1.0])
        result = apply_top_k_top_p(logits, None, p)
        # With p=1.0, cumulative sum will be <= 0 for nothing, so all survive
        # (at least one is guaranteed)
        non_inf = (result[0] != float("-inf")).sum().item()
        assert non_inf >= 1

    def test_top_k_and_top_p_combined(self):
        """Both k and p should be applied."""
        logits = torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0]])
        k = torch.tensor([3])
        p = torch.tensor([0.8])
        result = apply_top_k_top_p(logits, k, p)
        # Top-k limits to 3, then top-p further filters
        non_inf = (result[0] != float("-inf")).sum().item()
        assert 1 <= non_inf <= 3

    def test_batch_processing(self):
        """Should handle batch dimension correctly."""
        logits = torch.randn(4, 20)
        k = torch.tensor([5, 3, 10, 1])
        result = apply_top_k_top_p(logits, k, None)
        assert result.shape == logits.shape
        # Each row should have at most k non-inf values
        for i, ki in enumerate([5, 3, 10, 1]):
            non_inf = (result[i] != float("-inf")).sum().item()
            assert non_inf <= ki

    def test_top_p_at_least_one(self):
        """Top-p should always keep at least one token."""
        logits = torch.tensor([[10.0, -100.0, -100.0, -100.0]])
        p = torch.tensor([0.01])  # Very small p
        result = apply_top_k_top_p(logits, None, p)
        non_inf = (result[0] != float("-inf")).sum().item()
        assert non_inf >= 1
