# Copyright 2025 Rebellions Inc. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

from unittest.mock import MagicMock, patch

import pytest

from vllm_rbln.worker.utils import estimate_available_memory

pytestmark = pytest.mark.cpu_test


def _make_model_config(
    num_layers=12,
    head_dim=64,
    vocab_size=32000,
    hidden_size=768,
    num_kv_heads=8,
):
    cfg = MagicMock()
    cfg.get_num_layers.return_value = num_layers
    cfg.get_head_size.return_value = head_dim
    cfg.get_vocab_size.return_value = vocab_size
    cfg.get_hidden_size.return_value = hidden_size
    cfg.get_num_kv_heads.return_value = num_kv_heads
    return cfg


def _make_parallel_config(tp=1, dp=1, ep=False):
    cfg = MagicMock()
    cfg.tensor_parallel_size = tp
    cfg.data_parallel_size = dp
    cfg.enable_expert_parallel = ep
    return cfg


@patch("vllm_rbln.worker.utils.current_platform")
@patch("vllm_rbln.worker.utils.envs")
def test_atom_device(mock_envs, mock_platform):
    mock_platform.get_device_name.return_value = "RBLN-CA25"
    mock_envs.VLLM_RBLN_TP_SIZE = 1

    model_cfg = _make_model_config()
    parallel_cfg = _make_parallel_config()

    result = estimate_available_memory(
        model_cfg, parallel_cfg, kernel_size=1 * 2**30
    )
    assert result > 0


@patch("vllm_rbln.worker.utils.current_platform")
@patch("vllm_rbln.worker.utils.envs")
def test_rebel_device(mock_envs, mock_platform):
    mock_platform.get_device_name.return_value = "RBLN-CR100"
    mock_envs.VLLM_RBLN_TP_SIZE = 1

    model_cfg = _make_model_config()
    parallel_cfg = _make_parallel_config()

    result = estimate_available_memory(
        model_cfg, parallel_cfg, kernel_size=1 * 2**30
    )
    assert result > 0


@patch("vllm_rbln.worker.utils.current_platform")
@patch("vllm_rbln.worker.utils.envs")
def test_oom_raises(mock_envs, mock_platform):
    mock_platform.get_device_name.return_value = "RBLN-CA25"
    mock_envs.VLLM_RBLN_TP_SIZE = 1

    model_cfg = _make_model_config()
    parallel_cfg = _make_parallel_config()

    # kernel_size larger than available DRAM
    with pytest.raises(MemoryError, match="Insufficient DRAM"):
        estimate_available_memory(
            model_cfg, parallel_cfg, kernel_size=100 * 2**30
        )


@patch("vllm_rbln.worker.utils.current_platform")
@patch("vllm_rbln.worker.utils.envs")
def test_both_params_raises(mock_envs, mock_platform):
    mock_platform.get_device_name.return_value = "RBLN-CA25"
    mock_envs.VLLM_RBLN_TP_SIZE = 1

    model_cfg = _make_model_config()
    parallel_cfg = _make_parallel_config()

    with pytest.raises(ValueError, match="cannot be specified"):
        estimate_available_memory(
            model_cfg,
            parallel_cfg,
            kernel_size=1 * 2**30,
            n_model_params=100_000_000,
        )


@patch("vllm_rbln.worker.utils.current_platform")
@patch("vllm_rbln.worker.utils.envs")
def test_estimated_kernel_from_params(mock_envs, mock_platform):
    mock_platform.get_device_name.return_value = "RBLN-CA25"
    mock_envs.VLLM_RBLN_TP_SIZE = 1

    model_cfg = _make_model_config()
    parallel_cfg = _make_parallel_config()

    result = estimate_available_memory(
        model_cfg, parallel_cfg, n_model_params=100_000_000, nbits_per_param=16
    )
    assert result > 0


@patch("vllm_rbln.worker.utils.current_platform")
@patch("vllm_rbln.worker.utils.envs")
def test_invalid_device_raises(mock_envs, mock_platform):
    mock_platform.get_device_name.return_value = "UNKNOWN-DEVICE"
    mock_envs.VLLM_RBLN_TP_SIZE = 1

    model_cfg = _make_model_config()
    parallel_cfg = _make_parallel_config()

    with pytest.raises(AssertionError):
        estimate_available_memory(
            model_cfg, parallel_cfg, kernel_size=1 * 2**30
        )
