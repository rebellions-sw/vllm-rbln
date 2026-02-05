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
import pytest
import vllm_rbln.rbln_envs as rbln_envs


def test_rbln_envs():
    # check default values
    assert rbln_envs.VLLM_RBLN_COMPILE_MODEL, (
        f"Expected VLLM_RBLN_COMPILE_MODEL to be True, \
        got {rbln_envs.VLLM_RBLN_COMPILE_MODEL}"
    )

    assert not rbln_envs.VLLM_RBLN_COMPILE_STRICT_MODE, (
        f"Expected VLLM_RBLN_COMPILE_STRICT_MODE to be False, \
        got {rbln_envs.VLLM_RBLN_COMPILE_STRICT_MODE}"
    )

    assert rbln_envs.VLLM_RBLN_TP_SIZE == 1, (
        f"Expected VLLM_RBLN_TP_SIZE to be 1, \
        got {rbln_envs.VLLM_RBLN_TP_SIZE}"
    )

    assert rbln_envs.VLLM_RBLN_SAMPLER, (
        f"Expected VLLM_RBLN_SAMPLER to be True, \
        got {rbln_envs.VLLM_RBLN_SAMPLER}"
    )

    assert rbln_envs.VLLM_RBLN_ENABLE_WARM_UP, (
        f"Expected VLLM_RBLN_ENABLE_WARM_UP to be True, \
        got {rbln_envs.VLLM_RBLN_ENABLE_WARM_UP}"
    )

    assert rbln_envs.VLLM_RBLN_USE_VLLM_MODEL, (
        f"Expected VLLM_RBLN_USE_VLLM_MODEL to be True, \
        got {rbln_envs.VLLM_RBLN_USE_VLLM_MODEL}"
    )

    assert rbln_envs.VLLM_RBLN_FLASH_CAUSAL_ATTN, (
        f"Expected VLLM_RBLN_FLASH_CAUSAL_ATTN to be True, \
        got {rbln_envs.VLLM_RBLN_FLASH_CAUSAL_ATTN}"
    )

    assert not rbln_envs.VLLM_RBLN_DISABLE_MM, (
        f"Expected VLLM_RBLN_DISABLE_MM to be False, \
        got {rbln_envs.VLLM_RBLN_DISABLE_MM}"
    )

    assert rbln_envs.VLLM_RBLN_DP_IMPL == "padded_decode", (
        f"Expected VLLM_RBLN_DP_IMPL to be padded_decode, \
        got {rbln_envs.VLLM_RBLN_DP_IMPL}"
    )

    assert not rbln_envs.VLLM_RBLN_ENFORCE_MODEL_FP32, (
        f"Expected VLLM_RBLN_ENFORCE_MODEL_FP32 to be False, \
        got {rbln_envs.VLLM_RBLN_ENFORCE_MODEL_FP32}"
    )

    assert rbln_envs.VLLM_RBLN_MOE_CUSTOM_KERNEL, (
        f"Expected VLLM_RBLN_MOE_CUSTOM_KERNEL to be True, \
        got {rbln_envs.VLLM_RBLN_MOE_CUSTOM_KERNEL}"
    )

    assert rbln_envs.VLLM_RBLN_DP_INPUT_ALL_GATHER, (
        f"Expected VLLM_RBLN_DP_INPUT_ALL_GATHER to be True, \
        got {rbln_envs.VLLM_RBLN_DP_INPUT_ALL_GATHER}"
    )

    assert rbln_envs.VLLM_RBLN_LOGITS_ALL_GATHER, (
        f"Expected VLLM_RBLN_LOGITS_ALL_GATHER to be True, \
        got {rbln_envs.VLLM_RBLN_LOGITS_ALL_GATHER}"
    )

    assert rbln_envs.VLLM_RBLN_NUM_RAY_NODES == 1, (
        f"Expected VLLM_RBLN_NUM_RAY_NODES to be 1, \
        got {rbln_envs.VLLM_RBLN_NUM_RAY_NODES}"
    )

    assert not rbln_envs.VLLM_RBLN_METRICS, (
        f"Expected VLLM_RBLN_METRICS to be False, \
        got {rbln_envs.VLLM_RBLN_METRICS}"
    )


def test_get_dp_impl_default(monkeypatch):
    """Test get_dp_impl returns default when env var is not set."""
    monkeypatch.delenv("VLLM_RBLN_DP_IMPL", raising=False)
    from vllm_rbln.rbln_envs import get_dp_impl

    assert get_dp_impl() == "padded_decode"


def test_get_dp_impl_valid_choices(monkeypatch):
    """Test get_dp_impl with valid choices."""
    from vllm_rbln.rbln_envs import get_dp_impl

    monkeypatch.setenv("VLLM_RBLN_DP_IMPL", "padded_decode")
    assert get_dp_impl() == "padded_decode"

    monkeypatch.setenv("VLLM_RBLN_DP_IMPL", "dummy_prefill")
    assert get_dp_impl() == "dummy_prefill"

    # case insensitive
    monkeypatch.setenv("VLLM_RBLN_DP_IMPL", "PADDED_DECODE")
    assert get_dp_impl() == "padded_decode"


def test_get_dp_impl_invalid(monkeypatch):
    """Test get_dp_impl raises ValueError for invalid choice."""
    from vllm_rbln.rbln_envs import get_dp_impl

    monkeypatch.setenv("VLLM_RBLN_DP_IMPL", "invalid_impl")
    with pytest.raises(ValueError, match="Invalid VLLM_RBLN_DP_IMPL"):
        get_dp_impl()


def test_getattr_invalid():
    """Test __getattr__ raises AttributeError for unknown names."""
    with pytest.raises(AttributeError, match="has no attribute"):
        _ = rbln_envs.THIS_DOES_NOT_EXIST


def test_env_overrides(monkeypatch):
    """Test that env var overrides work via __getattr__."""
    monkeypatch.setenv("VLLM_RBLN_TP_SIZE", "4")
    assert rbln_envs.VLLM_RBLN_TP_SIZE == 4

    monkeypatch.setenv("VLLM_RBLN_COMPILE_MODEL", "false")
    assert not rbln_envs.VLLM_RBLN_COMPILE_MODEL

    monkeypatch.setenv("VLLM_RBLN_SORT_BATCH", "true")
    assert rbln_envs.VLLM_RBLN_SORT_BATCH
