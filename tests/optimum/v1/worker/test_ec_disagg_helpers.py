# Copyright 2026 Rebellions Inc. All rights reserved.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at:

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import types

import pytest

from vllm_rbln.v1.worker.ec_disagg_helpers import ECDisaggHelpersMixin
from vllm_rbln.v1.worker.optimum_model_runner import RBLNOptimumModelRunner


def _feature(identifier, offset, length):
    return types.SimpleNamespace(
        identifier=identifier,
        data="pixels",
        modality="image",
        mm_position=types.SimpleNamespace(offset=offset, length=length, is_embed=None),
    )


def _scheduler_output(features):
    return types.SimpleNamespace(
        scheduled_new_reqs=[types.SimpleNamespace(mm_features=features)]
    )


def _consumer(encoder_cache):
    """Fake runner: the helper mixin's ``_gather_cached_mm_outputs`` walks the
    runner's ``_iter_kept_mm_features`` and reads ``encoder_cache``."""
    fake = types.SimpleNamespace(
        encoder_cache=encoder_cache, is_multimodal_raw_input_only_model=True
    )
    fake._iter_kept_mm_features = types.MethodType(
        RBLNOptimumModelRunner._iter_kept_mm_features, fake
    )
    fake._gather_cached_mm_outputs = types.MethodType(
        ECDisaggHelpersMixin._gather_cached_mm_outputs, fake
    )
    return fake


class TestGatherCachedMmOutputs:
    FEATURES = [_feature("imgA", 15, 396), _feature("imgB", 413, 510)]

    def test_returns_every_item_in_prompt_order(self):
        consumer = _consumer({"imgB": "encB", "imgA": "encA"})
        out = consumer._gather_cached_mm_outputs(_scheduler_output(self.FEATURES), 0)
        assert out == ["encA", "encB"]

    def test_drops_items_fully_inside_the_cached_prefix(self):
        # imgA ends at 411; a boundary at 411 leaves only imgB to rebuild, the
        # same item _mm_embed_tail_starts keeps, so the two stay aligned.
        consumer = _consumer({"imgA": "encA", "imgB": "encB"})
        out = consumer._gather_cached_mm_outputs(
            _scheduler_output(self.FEATURES), num_cached_tokens=411
        )
        assert out == ["encB"]

    def test_missing_item_is_an_error(self):
        consumer = _consumer({"imgA": "encA"})
        with pytest.raises(RuntimeError, match="cache miss: mm_hash=imgB"):
            consumer._gather_cached_mm_outputs(_scheduler_output(self.FEATURES), 0)

    def test_text_only_prefill_gathers_nothing(self):
        consumer = _consumer({})
        assert consumer._gather_cached_mm_outputs(_scheduler_output([]), 0) == []
