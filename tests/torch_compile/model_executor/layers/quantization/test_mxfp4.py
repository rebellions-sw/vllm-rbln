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

from vllm_rbln.model_executor.layers.quantization.mxfp4 import (
    _dequantize_mxfp4,
    _swigluoai,
)


class TestDequantizeMxfp4:
    def test_output_shape(self):
        """Dequantized output should have 2x the packed dimension."""
        blocks = torch.zeros(4, 16, dtype=torch.uint8)
        scales = torch.full((4, 1), 127, dtype=torch.uint8)  # exponent 0
        result = _dequantize_mxfp4(blocks, scales, torch.float32)
        assert result.shape == (4, 32)  # 16 * 2

    def test_zero_blocks(self):
        """All-zero blocks with neutral scale should produce zeros."""
        blocks = torch.zeros(2, 16, dtype=torch.uint8)
        # Scale = 127 means exponent = 0, so 2^0 = 1.0
        scales = torch.full((2, 1), 127, dtype=torch.uint8)
        result = _dequantize_mxfp4(blocks, scales, torch.float32)
        # FP4 value for nibble 0x0 = +0.0, so all should be zero
        assert torch.all(result == 0.0)

    def test_known_nibble_values(self):
        """Verify specific FP4 nibble lookups."""
        # nibble 0x1 = 0.5, nibble 0x2 = 1.0
        # packed: lo=0x1, hi=0x2 -> byte = (0x2 << 4) | 0x1 = 0x21
        blocks = torch.tensor([[0x21]], dtype=torch.uint8)
        scales = torch.full((1, 1), 127, dtype=torch.uint8)  # 2^0 = 1.0
        result = _dequantize_mxfp4(blocks, scales, torch.float32)
        assert result.shape == (1, 2)
        assert abs(result[0, 0].item() - 0.5) < 1e-6  # lo nibble = 0x1 -> 0.5
        assert abs(result[0, 1].item() - 1.0) < 1e-6  # hi nibble = 0x2 -> 1.0

    def test_scale_exponent(self):
        """Scale should multiply by 2^(scale - 127)."""
        # nibble 0x2 = 1.0, with scale = 128 -> 2^1 = 2.0
        blocks = torch.tensor(
            [[0x22]], dtype=torch.uint8
        )  # both nibbles = 0x2 = 1.0
        scales = torch.full((1, 1), 128, dtype=torch.uint8)  # 2^1 = 2.0
        result = _dequantize_mxfp4(blocks, scales, torch.float32)
        # 1.0 * 2.0 = 2.0
        assert abs(result[0, 0].item() - 2.0) < 1e-6
        assert abs(result[0, 1].item() - 2.0) < 1e-6

    def test_negative_fp4_values(self):
        """Nibbles >= 0x8 should produce negative values."""
        # nibble 0xA = -1.0 (index 10 in LUT)
        # packed: lo=0xA, hi=0x0 -> byte = 0x0A
        blocks = torch.tensor([[0x0A]], dtype=torch.uint8)
        scales = torch.full((1, 1), 127, dtype=torch.uint8)
        result = _dequantize_mxfp4(blocks, scales, torch.float32)
        assert abs(result[0, 0].item() - (-1.0)) < 1e-6  # lo = 0xA -> -1.0
        assert abs(result[0, 1].item() - 0.0) < 1e-6  # hi = 0x0 -> +0.0

    def test_batched(self):
        """Should work with higher-dimensional inputs."""
        blocks = torch.zeros(2, 3, 16, dtype=torch.uint8)
        scales = torch.full((2, 3, 1), 127, dtype=torch.uint8)
        result = _dequantize_mxfp4(blocks, scales, torch.float32)
        assert result.shape == (2, 3, 32)


class TestSwigluoai:
    def test_basic(self):
        """swigluoai should compute (up + 1) * gate * sigmoid(gate * alpha)."""
        gate = torch.tensor([0.0])
        up = torch.tensor([0.0])
        alpha = 1.702
        limit = 7.0
        result = _swigluoai(gate, up, alpha, limit)
        # gate=0 -> sigmoid(0) = 0.5, glu = 0 * 0.5 = 0
        # (up+1) * glu = 1 * 0 = 0
        assert abs(result.item()) < 1e-6

    def test_positive_gate(self):
        """Positive gate should produce positive output when up >= -1."""
        gate = torch.tensor([2.0])
        up = torch.tensor([1.0])
        result = _swigluoai(gate, up, 1.702, 7.0)
        assert result.item() > 0

    def test_clamp_limit(self):
        """Gate should be clamped to limit, up clamped to [-limit, limit]."""
        gate = torch.tensor([100.0])
        up = torch.tensor([100.0])
        result = _swigluoai(gate, up, 1.702, 7.0)
        # gate clamped to 7.0, up clamped to 7.0
        expected_gate = torch.tensor([7.0])
        expected_up = torch.tensor([7.0])
        expected = (
            (expected_up + 1)
            * expected_gate
            * torch.sigmoid(expected_gate * 1.702)
        )
        assert torch.allclose(result, expected, atol=1e-5)

    def test_negative_up(self):
        """up is clamped to [-limit, limit]."""
        gate = torch.tensor([1.0])
        up = torch.tensor([-100.0])
        result = _swigluoai(gate, up, 1.702, 7.0)
        # up clamped to -7.0
        expected_up = torch.tensor([-7.0])
        expected_gate = torch.tensor([1.0])
        glu = expected_gate * torch.sigmoid(expected_gate * 1.702)
        expected = (expected_up + 1) * glu
        assert torch.allclose(result, expected, atol=1e-5)

    def test_batch(self):
        """Should work on batched inputs."""
        gate = torch.randn(4, 8)
        up = torch.randn(4, 8)
        result = _swigluoai(gate, up, 1.702, 7.0)
        assert result.shape == (4, 8)
