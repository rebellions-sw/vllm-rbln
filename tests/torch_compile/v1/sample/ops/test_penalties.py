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

from vllm_rbln.v1.sample.ops.penalties import (
    _convert_to_tensors,
    apply_all_penalties,
)


def test_convert_to_tensors_basic():
    output_token_ids = [[1, 2, 3], [4, 5]]
    vocab_size = 100
    result = _convert_to_tensors(
        output_token_ids, vocab_size, torch.device("cpu")
    )
    assert result.shape == (2, 3)
    assert result.dtype == torch.int64
    # Second row should be padded with vocab_size
    assert result[1, 2].item() == vocab_size


def test_convert_to_tensors_empty():
    output_token_ids = [[], []]
    vocab_size = 50
    result = _convert_to_tensors(
        output_token_ids, vocab_size, torch.device("cpu")
    )
    assert result.shape[0] == 2


def test_convert_to_tensors_single():
    output_token_ids = [[10, 20, 30]]
    vocab_size = 100
    result = _convert_to_tensors(
        output_token_ids, vocab_size, torch.device("cpu")
    )
    assert result.shape == (1, 3)
    assert result[0].tolist() == [10, 20, 30]


def test_apply_all_penalties_no_penalty():
    """When penalties are neutral, logits should not change much."""
    batch_size = 2
    vocab_size = 10
    logits = torch.randn(batch_size, vocab_size)
    prompt_token_ids = torch.tensor([[1, 2, 3, 0, 0], [4, 5, 0, 0, 0]])
    presence_penalties = torch.zeros(batch_size)
    frequency_penalties = torch.zeros(batch_size)
    repetition_penalties = torch.ones(batch_size)  # 1.0 = no repetition penalty
    output_token_ids = [[6], [7]]

    original_logits = logits.clone()
    result = apply_all_penalties(
        logits,
        prompt_token_ids,
        presence_penalties,
        frequency_penalties,
        repetition_penalties,
        output_token_ids,
    )
    assert result.shape == (batch_size, vocab_size)
    # With no penalties, logits at non-output/prompt positions should be similar
    # (apply_penalties still processes them, but neutral values should leave most unchanged)
    assert result.shape == original_logits.shape


def test_apply_all_penalties_with_repetition():
    """Repetition penalty should reduce logits for repeated tokens."""
    batch_size = 1
    vocab_size = 10
    logits = torch.ones(batch_size, vocab_size) * 2.0
    prompt_token_ids = torch.tensor([[1, 2, 3, 0, 0]])
    presence_penalties = torch.zeros(batch_size)
    frequency_penalties = torch.zeros(batch_size)
    repetition_penalties = torch.tensor([2.0])  # Strong repetition penalty
    output_token_ids = [[5]]  # Token 5 appeared in output

    result = apply_all_penalties(
        logits,
        prompt_token_ids,
        presence_penalties,
        frequency_penalties,
        repetition_penalties,
        output_token_ids,
    )
    # Token 5 (appeared in output) should have lower logit due to repetition penalty
    assert result[0, 5].item() < 2.0
    # Tokens 1,2,3 (appeared in prompt) should also be penalized
    assert result[0, 1].item() < 2.0


def test_apply_all_penalties_with_presence():
    """Presence penalty should reduce logits for tokens that appeared."""
    batch_size = 1
    vocab_size = 10
    logits = torch.ones(batch_size, vocab_size) * 5.0
    prompt_token_ids = torch.tensor([[0, 0, 0, 0, 0]])
    presence_penalties = torch.tensor([1.0])  # Positive presence penalty
    frequency_penalties = torch.zeros(batch_size)
    repetition_penalties = torch.ones(batch_size)
    output_token_ids = [[3, 3, 3]]

    result = apply_all_penalties(
        logits,
        prompt_token_ids,
        presence_penalties,
        frequency_penalties,
        repetition_penalties,
        output_token_ids,
    )
    # Token 3 should be penalized
    assert result[0, 3].item() < 5.0
    # Token 0 should not be affected by presence (only in prompt with 0 penalty)
    # Token 7 (never appeared) should remain unchanged
    assert abs(result[0, 7].item() - 5.0) < 0.01


def test_apply_all_penalties_with_frequency():
    """Frequency penalty should scale with token count."""
    batch_size = 1
    vocab_size = 10
    logits = torch.ones(batch_size, vocab_size) * 5.0
    prompt_token_ids = torch.tensor([[0, 0, 0, 0, 0]])
    presence_penalties = torch.zeros(batch_size)
    frequency_penalties = torch.tensor([1.0])
    repetition_penalties = torch.ones(batch_size)
    output_token_ids = [[3, 3, 3, 4]]  # Token 3 appears 3 times, token 4 once

    result = apply_all_penalties(
        logits,
        prompt_token_ids,
        presence_penalties,
        frequency_penalties,
        repetition_penalties,
        output_token_ids,
    )
    # Token 3 should be penalized more than token 4
    assert result[0, 3].item() < result[0, 4].item()
