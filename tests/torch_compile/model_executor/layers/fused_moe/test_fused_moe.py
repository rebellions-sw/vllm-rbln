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

from unittest.mock import MagicMock, patch

import torch

from vllm_rbln.model_executor.layers.fused_moe.layer import (
    get_masked_routing_weights,
    unquantized_fused_moe_method_rbln,
)


class TestGetMaskedRoutingWeights:
    def test_basic_routing(self):
        """Basic top-k routing should produce correct masked weights."""
        with patch(
            "vllm_rbln.model_executor.layers.fused_moe.layer.envs"
        ) as mock_envs:
            mock_envs.VLLM_RBLN_USE_MOE_TOKENS_MASK = False

            router_logits = torch.tensor(
                [
                    [2.0, 1.0, 0.5, 0.1],  # token 0: expert 0 > 1 > 2 > 3
                ]
            )
            masked_weights, expert_count = get_masked_routing_weights(
                router_logits, top_k=2, renormalize=False, expert_map=None
            )
            assert masked_weights.shape == (1, 4)
            # Top-2 experts (0, 1) should have non-zero weights
            assert masked_weights[0, 0].item() > 0
            assert masked_weights[0, 1].item() > 0
            assert masked_weights[0, 2].item() == 0
            assert masked_weights[0, 3].item() == 0

    def test_expert_select_count(self):
        """expert_select_count should count how many tokens each expert was selected for."""
        with patch(
            "vllm_rbln.model_executor.layers.fused_moe.layer.envs"
        ) as mock_envs:
            mock_envs.VLLM_RBLN_USE_MOE_TOKENS_MASK = False

            router_logits = torch.tensor(
                [
                    [5.0, 1.0, 0.0, 0.0],  # token 0: expert 0, 1
                    [0.0, 5.0, 1.0, 0.0],  # token 1: expert 1, 2
                    [0.0, 0.0, 5.0, 1.0],  # token 2: expert 2, 3
                ]
            )
            _, expert_count = get_masked_routing_weights(
                router_logits, top_k=2, renormalize=False, expert_map=None
            )
            assert expert_count.shape == (4,)
            assert expert_count[0].item() == 1  # expert 0: 1 token
            assert expert_count[1].item() == 2  # expert 1: 2 tokens
            assert expert_count[2].item() == 2  # expert 2: 2 tokens
            assert expert_count[3].item() == 1  # expert 3: 1 token

    def test_renormalize(self):
        """With renormalize=True, routing weights should sum to 1 per token."""
        with patch(
            "vllm_rbln.model_executor.layers.fused_moe.layer.envs"
        ) as mock_envs:
            mock_envs.VLLM_RBLN_USE_MOE_TOKENS_MASK = False

            router_logits = torch.tensor([[3.0, 2.0, 1.0, 0.0]])
            masked_weights, _ = get_masked_routing_weights(
                router_logits, top_k=2, renormalize=True, expert_map=None
            )
            # Sum of non-zero weights should be ~1.0
            total = masked_weights[0].sum().item()
            assert abs(total - 1.0) < 1e-5

    def test_expert_map(self):
        """expert_map should remap selected expert indices."""
        with patch(
            "vllm_rbln.model_executor.layers.fused_moe.layer.envs"
        ) as mock_envs:
            mock_envs.VLLM_RBLN_USE_MOE_TOKENS_MASK = False

            router_logits = torch.tensor([[5.0, 0.0, 0.0, 3.0]])
            # Map: global expert 0 -> local 2, global expert 3 -> local 1
            expert_map = torch.tensor([2, -1, -1, 1], dtype=torch.int64)
            masked_weights, _ = get_masked_routing_weights(
                router_logits, top_k=2, renormalize=False, expert_map=expert_map
            )
            # After mapping, experts at indices 2 and 1 should have weights
            assert masked_weights[0, 2].item() > 0  # mapped from expert 0
            assert masked_weights[0, 1].item() > 0  # mapped from expert 3


class TestUnquantizedFusedMoeMethodRbln:
    def test_basic_forward(self):
        """Basic forward pass through the native PyTorch MoE implementation."""
        num_experts = 2
        hidden_size = 8
        intermediate_size = 4
        top_k = 1
        num_tokens = 2

        mock_self = MagicMock()
        mock_layer = MagicMock()

        # w13_weight: [num_experts, 2 * intermediate_size, hidden_size]
        mock_layer.w13_weight = torch.randn(
            num_experts, 2 * intermediate_size, hidden_size
        )
        # w2_weight: [num_experts, hidden_size, intermediate_size]
        mock_layer.w2_weight = torch.randn(
            num_experts, hidden_size, intermediate_size
        )

        x = torch.randn(num_tokens, hidden_size)
        router_logits = torch.tensor(
            [
                [5.0, 0.0],  # token 0 -> expert 0
                [0.0, 5.0],  # token 1 -> expert 1
            ]
        )

        result = unquantized_fused_moe_method_rbln(
            mock_self,
            mock_layer,
            x,
            use_grouped_topk=False,
            top_k=top_k,
            router_logits=router_logits,
            renormalize=True,
        )
        assert result.shape == x.shape

    def test_preserves_shape(self):
        """Output shape should match input shape."""
        num_experts = 4
        hidden_size = 16
        intermediate_size = 8
        batch_size = 3

        mock_self = MagicMock()
        mock_layer = MagicMock()
        mock_layer.w13_weight = torch.randn(
            num_experts, 2 * intermediate_size, hidden_size
        )
        mock_layer.w2_weight = torch.randn(
            num_experts, hidden_size, intermediate_size
        )

        x = torch.randn(batch_size, hidden_size)
        router_logits = torch.randn(batch_size, num_experts)

        result = unquantized_fused_moe_method_rbln(
            mock_self,
            mock_layer,
            x,
            use_grouped_topk=False,
            top_k=2,
            router_logits=router_logits,
            renormalize=True,
        )
        assert result.shape == x.shape
