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

"""Tests for rbln_model_loader.

NOTE: rbln_model_loader imports from optimum which may not be available
in all environments. Tests handle import errors gracefully.
"""

import importlib
import sys
from unittest.mock import MagicMock

import pytest


class TestGetOptimumModel:
    """Test get_optimum_model function."""

    def test_delegates_to_load_model(self):
        """get_optimum_model should delegate to optimum load_model."""
        # The module imports load_model from vllm_rbln.model_executor.models.optimum
        # We need to mock that at module level before importing
        fake_model = MagicMock()
        mock_optimum = MagicMock()
        mock_optimum.load_model = MagicMock(return_value=fake_model)

        # Temporarily inject the mock module
        mod_key = "vllm_rbln.model_executor.models.optimum"
        loader_key = "vllm_rbln.model_executor.model_loader.rbln_model_loader"

        # Remove cached module if present
        old_mod = sys.modules.pop(loader_key, None)
        old_optimum = sys.modules.get(mod_key)
        sys.modules[mod_key] = mock_optimum

        try:
            mod = importlib.import_module(loader_key)
            importlib.reload(mod)  # Force reload with mocked dependency

            vllm_config = MagicMock()
            result = mod.get_optimum_model(vllm_config)
            mock_optimum.load_model.assert_called_once_with(vllm_config)
            assert result is fake_model
        finally:
            # Restore original state
            if old_mod is not None:
                sys.modules[loader_key] = old_mod
            else:
                sys.modules.pop(loader_key, None)
            if old_optimum is not None:
                sys.modules[mod_key] = old_optimum
            else:
                sys.modules.pop(mod_key, None)
