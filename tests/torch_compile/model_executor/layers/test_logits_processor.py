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

from vllm_rbln.model_executor.layers.logits_processor import (
    logits_processor_gather_logits,
    logits_processor_get_logits,
)


def test_get_logits_basic():
    """get_logits should call lm_head.quant_method.apply."""
    mock_self = MagicMock()
    hidden_states = torch.randn(2, 4, 128)
    lm_head = MagicMock()
    expected = torch.randn(2, 4, 1000)
    lm_head.quant_method.apply.return_value = expected

    result = logits_processor_get_logits(
        mock_self, hidden_states, lm_head, None
    )

    lm_head.quant_method.apply.assert_called_once_with(
        lm_head, hidden_states, bias=None
    )
    assert torch.equal(result, expected)


def test_get_logits_with_bias():
    """get_logits should pass embedding_bias to apply."""
    mock_self = MagicMock()
    hidden_states = torch.randn(1, 1, 64)
    lm_head = MagicMock()
    bias = torch.randn(1000)
    expected = torch.randn(1, 1, 1000)
    lm_head.quant_method.apply.return_value = expected

    result = logits_processor_get_logits(
        mock_self, hidden_states, lm_head, bias
    )

    lm_head.quant_method.apply.assert_called_once_with(
        lm_head, hidden_states, bias=bias
    )
    assert torch.equal(result, expected)


def test_gather_logits_all_gather():
    """When use_all_gather=True, should call tensor_model_parallel_all_gather."""
    mock_self = MagicMock()
    mock_self.use_all_gather = True
    mock_self.org_vocab_size = 50
    logits = torch.randn(2, 100)

    with patch(
        "vllm_rbln.model_executor.layers.logits_processor.tensor_model_parallel_all_gather"
    ) as mock_all_gather:
        mock_all_gather.return_value = logits
        result = logits_processor_gather_logits(mock_self, logits)
        mock_all_gather.assert_called_once_with(logits)
    # Should be trimmed to org_vocab_size
    assert result.shape[-1] == 50


def test_gather_logits_gather():
    """When use_all_gather=False, should call tensor_model_parallel_gather."""
    mock_self = MagicMock()
    mock_self.use_all_gather = False
    mock_self.org_vocab_size = 30
    logits = torch.randn(2, 50)

    with patch(
        "vllm_rbln.model_executor.layers.logits_processor.tensor_model_parallel_gather"
    ) as mock_gather:
        mock_gather.return_value = logits
        result = logits_processor_gather_logits(mock_self, logits)
        mock_gather.assert_called_once_with(logits)
    assert result.shape[-1] == 30


def test_gather_logits_none_for_non_rank0():
    """gather may return None for non-rank-0 processes."""
    mock_self = MagicMock()
    mock_self.use_all_gather = False
    mock_self.org_vocab_size = 50

    with patch(
        "vllm_rbln.model_executor.layers.logits_processor.tensor_model_parallel_gather"
    ) as mock_gather:
        mock_gather.return_value = None
        result = logits_processor_gather_logits(mock_self, torch.randn(2, 100))
    assert result is None
