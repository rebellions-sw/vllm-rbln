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

from vllm_rbln.v1.core.rbln_scheduler import is_prefill


def test_is_prefill_true():
    """Request with computed < total - 1 is in prefill."""
    request = MagicMock()
    request.num_computed_tokens = 0
    request.num_tokens = 100
    assert is_prefill(request) is True


def test_is_prefill_partially_computed():
    """Request with some tokens computed but not all is still prefill."""
    request = MagicMock()
    request.num_computed_tokens = 50
    request.num_tokens = 100
    assert is_prefill(request) is True


def test_is_prefill_false_decode():
    """Request with computed == total - 1 is in decode (not prefill)."""
    request = MagicMock()
    request.num_computed_tokens = 99
    request.num_tokens = 100
    assert is_prefill(request) is False


def test_is_prefill_false_fully_computed():
    """Request with computed >= total - 1 is in decode."""
    request = MagicMock()
    request.num_computed_tokens = 100
    request.num_tokens = 100
    assert is_prefill(request) is False


def test_is_prefill_single_token():
    """Single token request: computed=0, total=1 -> not prefill (0 < 0 is False)."""
    request = MagicMock()
    request.num_computed_tokens = 0
    request.num_tokens = 1
    assert is_prefill(request) is False
