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
from vllm.platforms import current_platform

from .utils import (_schedule_cached_reqs, _schedule_new_request,
                    create_grammar_bitmask, create_model_runner, make_request)

DEVICE = current_platform.device_type


@pytest.mark.parametrize(
    "num_seqs, expected_bucket_sizes",
    [
        pytest.param(1, [1], id="1_seq"),
        pytest.param(2, [1, 2], id="2_seq"),
        pytest.param(4, [1, 2, 4], id="4_seq"),
        pytest.param(8, [1, 2, 4, 8], id="8_seq"),
        pytest.param(16, [1, 2, 4, 8, 16], id="16_seq"),
        # pytest.param(32, [1, 2, 4, 8, 16, 32], id="32_seq"),
        pytest.param(64, [1, 2, 4, 8, 16, 24, 32, 40, 48, 56, 64],
                     id="64_seq"),
        pytest.param(129, [
            1, 2, 4, 8, 16, 24, 32, 40, 48, 56, 64, 72, 80, 88, 96, 104, 112,
            120, 128
        ],
                     id="129_seq"),
        # pytest.param(256, [1, 2, 4, 8, 16, 32, 64, 128, 256], id="256_seq"),
        pytest.param(512, [
            1, 2, 4, 8, 16, 24, 32, 40, 48, 56, 64, 72, 80, 88, 96, 104, 112,
            120, 128, 136, 144, 152, 160, 168, 176, 184, 192, 200, 208, 216,
            224, 232, 240, 248, 256, 272, 288, 304, 320, 336, 352, 368, 384,
            400, 416, 432, 448, 464, 480, 496, 512
        ],
                     id="512_seq"),
    ])
def test_get_bucket_sizes(monkeypatch, num_seqs: int,
                          expected_bucket_sizes: list[int]):
    monkeypatch.setenv("VLLM_RBLN_SAMPLER", "1")
    runner = create_model_runner(max_num_seqs=num_seqs)
    bucket_sizes = runner.get_bucket_sizes(num_seqs)
    assert bucket_sizes == expected_bucket_sizes
    assert len(runner.pooled_tensors) == len(expected_bucket_sizes)


@pytest.mark.parametrize("use_rbln_sampler, use_structured_output", [
    pytest.param(True, True, id="use_rbln_sampler_and_structured_output"),
    pytest.param(True, False, id="use_rbln_sampler_and_no_structured_output"),
    pytest.param(False, True, id="no_rbln_sampler_and_structured_output"),
    pytest.param(False, False, id="no_rbln_sampler_and_no_structured_output"),
])
def test_forward_decode(monkeypatch, use_rbln_sampler, use_structured_output):
    """Test sampler logic for both use_rbln_sampler=True and False."""
    monkeypatch.setenv("VLLM_RBLN_SAMPLER", "1" if use_rbln_sampler else "0")
    runner = create_model_runner(max_num_seqs=4)

    req_ids = [f"req_{i}" for i in range(3)]
    reqs = []
    # Prefill
    for i in range(3):
        req_id = f"req_{i}"
        reqs.append(make_request(request_id=req_id, prompt_token_ids=[1, 2,
                                                                      3]))

    for i, req in enumerate(reqs):
        req_id = req.request_id
        scheduler_output = _schedule_new_request(req_id,
                                                 block_ids=([i], ),
                                                 outer_block_ids=[i])
        if use_structured_output:
            vocab_size = runner.model_config.get_vocab_size()
            scheduler_output.structured_output_request_ids = {req_id: i}
            scheduler_output.grammar_bitmask = create_grammar_bitmask(
                1, vocab_size)
        runner_output = runner.execute_model(scheduler_output)
        assert runner_output is not None
        assert runner_output.req_ids == [req_id]
        assert len(runner_output.sampled_token_ids) == 1

    # Update requests
    for i, req in enumerate(reqs):
        req.num_computed_tokens = 3

    # Decode
    scheduler_output = _schedule_cached_reqs(reqs,
                                             new_block_ids=[None, None, None])
    if use_structured_output:
        vocab_size = runner.model_config.get_vocab_size()
        scheduler_output.structured_output_request_ids = {
            req_id: i
            for i, req_id in enumerate(req_ids)
        }
        scheduler_output.grammar_bitmask = create_grammar_bitmask(
            3, vocab_size)

    runner_output = runner.execute_model(scheduler_output)

    assert runner_output is not None
    # req2 remains, and req0 and req1 are newly allocated in input_batch.req_ids
    assert runner_output.req_ids == ["req_2", "req_0", "req_1"]
    assert len(runner_output.sampled_token_ids) == 3
