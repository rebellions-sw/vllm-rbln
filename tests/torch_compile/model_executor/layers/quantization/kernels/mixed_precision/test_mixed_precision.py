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

from unittest.mock import MagicMock

from vllm.scalar_type import scalar_types

from vllm_rbln.model_executor.layers.quantization.kernels.mixed_precision.unpacked import (
    RBLNInt8UnpackedLinearKernel,
)


class TestRBLNInt8UnpackedLinearKernel:
    def _make_config(
        self,
        weight_type=None,
        zero_points=False,
        group_size=128,
        has_g_idx=False,
    ):
        config = MagicMock()
        config.weight_type = (
            scalar_types.uint4b8 if weight_type is None else weight_type
        )
        config.zero_points = zero_points
        config.group_size = group_size
        config.has_g_idx = has_g_idx
        return config

    def test_can_implement_uint4b8(self):
        """uint4b8 with valid config should be implementable."""
        config = self._make_config(weight_type=scalar_types.uint4b8)
        can, reason = RBLNInt8UnpackedLinearKernel.can_implement(config)
        assert can is True
        assert reason is None

    def test_can_implement_uint8b128(self):
        """uint8b128 should also be supported."""
        config = self._make_config(weight_type=scalar_types.uint8b128)
        can, reason = RBLNInt8UnpackedLinearKernel.can_implement(config)
        assert can is True

    def test_cannot_implement_unsupported_type(self):
        """Unsupported weight types should be rejected."""
        config = self._make_config(weight_type=scalar_types.float8_e4m3fn)
        can, reason = RBLNInt8UnpackedLinearKernel.can_implement(config)
        assert can is False
        assert "not supported" in reason

    def test_cannot_implement_zero_points(self):
        """Asymmetric quantization (zero_points=True) not supported."""
        config = self._make_config(zero_points=True)
        can, reason = RBLNInt8UnpackedLinearKernel.can_implement(config)
        assert can is False
        assert "Asymmetric" in reason

    def test_cannot_implement_bad_group_size(self):
        """Group sizes other than -1, 64, 128 should be rejected."""
        config = self._make_config(group_size=32)
        can, reason = RBLNInt8UnpackedLinearKernel.can_implement(config)
        assert can is False
        assert "Group size" in reason

    def test_can_implement_group_size_64(self):
        """Group size 64 should be accepted."""
        config = self._make_config(group_size=64)
        can, reason = RBLNInt8UnpackedLinearKernel.can_implement(config)
        assert can is True

    def test_can_implement_group_size_neg1(self):
        """Group size -1 (per-channel) should be accepted."""
        config = self._make_config(group_size=-1)
        can, reason = RBLNInt8UnpackedLinearKernel.can_implement(config)
        assert can is True

    def test_cannot_implement_g_idx(self):
        """Group/dynamic activation ordering not supported."""
        config = self._make_config(has_g_idx=True)
        can, reason = RBLNInt8UnpackedLinearKernel.can_implement(config)
        assert can is False
        assert "ordering" in reason
