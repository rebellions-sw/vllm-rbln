import torch
import pytest

from vllm_rbln.v1.worker.optimum_model_runner import RBLNOptimumModelRunner
from .utils import get_vllm_config, initialize_kv_cache, MockModelWrapper, _schedule_new_request, fake_load_model, create_model_runner
from vllm.config import (CacheConfig, ModelConfig, SchedulerConfig, VllmConfig,
                         set_current_vllm_config)
from vllm.distributed import (ensure_model_parallel_initialized,
                              init_distributed_environment)
import tempfile
import os
from vllm.platforms import current_platform
DEVICE = current_platform.device_type

# def fake_load_model(runner: RBLNOptimumModelRunner, num_reqs: int,
#                     vocab_size: int):
#     def fake_forward(self, model_input: ModelInputForRBLN,
#                 **kwargs) -> torch.Tensor:

#         return torch.randn((num_reqs, vocab_size), dtype=torch.float32)

#     runner.model = MockModelWrapper()
#     runner.use_optimum_lora = False
#     runner.model.forward = fake_forward(num_reqs, vocab_size)
#     # runner.compute_logits = fake_compute_logits(num_reqs, vocab_size)


@pytest.fixture
def model_runner_with_rbln_sampler(monkeypatch):
    """Fixture for model runner with RBLN sampler enabled."""
    monkeypatch.setenv("VLLM_RBLN_SAMPLER", "1")
    # vllm_config = get_vllm_config()
    # with set_current_vllm_config(vllm_config, check_compile=False):
    #     temp_file = tempfile.mkstemp()[1]
    #     init_distributed_environment(
    #         world_size=1,
    #         rank=0,
    #         local_rank=0,
    #         distributed_init_method=f"file://{temp_file}",
    #         backend="gloo",
    #     )
    #     ensure_model_parallel_initialized(
    #         1,
    #         1,
    #     )
    # runner = RBLNOptimumModelRunner(vllm_config, DEVICE)
    # initialize_kv_cache(runner)
    # fake_load_model(runner)
    # return runner


# @pytest.fixture
# def model_runner_without_rbln_sampler(monkeypatch):
#     """Fixture for model runner with RBLN sampler disabled."""
#     monkeypatch.setenv("VLLM_RBLN_SAMPLER", "0")
#     # runner = create_model_runner()
#     # return runner

@pytest.mark.parametrize("num_seqs, expected_bucket_sizes",
    [
        pytest.param(1, [1], id="1_seq"),
        pytest.param(2, [1, 2], id="2_seq"),
        pytest.param(4, [1, 2, 4], id="4_seq"),
        pytest.param(8, [1, 2, 4, 8], id="8_seq"),
        pytest.param(16, [1, 2, 4, 8, 16], id="16_seq"),
        # pytest.param(32, [1, 2, 4, 8, 16, 32], id="32_seq"),
        pytest.param(64, [1, 2, 4, 8, 16, 24, 32, 40, 48, 56, 64], id="64_seq"),
        pytest.param(129, [1, 2, 4, 8, 16, 24, 32, 40, 48, 56, 64, 72, 80, 88, 96, 104, 112, 120, 128], id="129_seq"),
        # pytest.param(256, [1, 2, 4, 8, 16, 32, 64, 128, 256], id="256_seq"),
        pytest.param(512, [1, 2, 4, 8, 16, 24, 32, 40, 48, 56, 64, 72, 80, 88, 96, 104, 112, 120, 128, 136, 144, 152, 160, 168, 176, 184, 192, 200, 208, 216, 224, 232, 240, 248, 256, 272, 288, 304, 320, 336, 352, 368, 384, 400, 416, 432, 448, 464, 480, 496, 512], id="512_seq"),
    ]
)
def test_get_bucket_sizes(monkeypatch, num_seqs: int, expected_bucket_sizes: list[int]):
    monkeypatch.setenv("VLLM_RBLN_SAMPLER", "1")
    runner = create_model_runner(max_num_seqs=num_seqs)
    bucket_sizes = runner.get_bucket_sizes(num_seqs)
    assert bucket_sizes == expected_bucket_sizes
    assert len(runner.pooled_tensors) == len(expected_bucket_sizes)

@pytest.mark.parametrize("expected_use_rbln_sampler", [True, False])
def test_sampler_with_rbln_sampler(monkeypatch, expected_use_rbln_sampler):
    """Test sampler logic for both use_rbln_sampler=True and False."""
    # 파라미터에 따라 환경 변수 설정
    monkeypatch.setenv("VLLM_RBLN_SAMPLER", "1" if expected_use_rbln_sampler else "0")
    runner = create_model_runner()

    # Schedule a request to set up the input batch
    scheduler_output = _schedule_new_request("req_0")
    # Execute model to trigger the sampler logic
    runner_output = runner.execute_model(scheduler_output)

    # Verify sampler_output was sliced correctly
    assert runner_output is not None
    assert runner_output.req_ids == ["req_0"]
    assert len(runner_output.sampled_token_ids) == 1
    # assert len(runner_output.logprobs) == 1

