from vllm.lora.request import LoRARequest
from vllm.lora.models import LoRAMapping

from vllm_rbln.v1.worker.optimum_worker import RBLNOptimumWorker as Worker
from vllm_rbln.worker.optimum_worker import RBLNOptimumWorker as V1Worker
from vllm.entrypoints.openai.api_server import (
    build_async_engine_client_from_engine_args)
from vllm.utils import merge_async_iterators
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm_rbln.v1.worker.optimum_model_runner import RBLNOptimumModelRunner
from vllm_rbln.worker.optimum_model_runner import RBLNOptimumModelRunner
from vllm_rbln.model_executor.models.optimum import RBLNOptimumForCausalLM
from vllm_rbln.model_executor.models.optimum.model_base import KVCacheBlockAdapter
from vllm import SamplingParams, TextPrompt
from vllm.config import ModelConfig, SchedulerConfig, CacheConfig, LoRAConfig, VllmConfig
import pytest
import asyncio
import torch

from unittest.mock import MagicMock, patch


NUM_LORAS = 5
BLOCK_SIZE = 16
NUM_BLOCKS = 8

def get_vllm_config(async_scheduling=False):
    model_config = ModelConfig(
        "meta-llama/Llama-2-7b-hf",
        dtype=torch.float32,
        seed=42,
    )
    scheduler_config = SchedulerConfig(
        max_num_seqs=10,
        max_num_batched_tokens=128,
        max_model_len=128,
        async_scheduling=async_scheduling,
    )
    cache_config = CacheConfig(
        block_size=BLOCK_SIZE,
        swap_space=0,
        cache_dtype="auto",
    )
    lora_config = LoRAConfig(max_lora_rank=8,
                            max_cpu_loras=NUM_LORAS,
                            max_loras=NUM_LORAS)
    vllm_config = VllmConfig(
        model_config=model_config,
        scheduler_config=scheduler_config,
        cache_config=cache_config,
        lora_config=lora_config,
    )
    return vllm_config

# def test_list_lora():


# def test_worker_apply_lora():
#     # lora_requests = [
#     #     LoRARequest(str(i + 1), i + 1, "/path/adapter" + str(i + 1))
#     #     for i in range(NUM_LORAS)
#     # ]
#     # lora_mapping = LoRAMapping(is_prefill=is_prompt,
#     #                             index_mapping=[],
#     #                             prompt_mapping=[])

#     worker_cls = V1Worker if envs.VLLM_USE_V1 else Worker
#     worker = worker_cls(
#         vllm_config=vllm_config,
#         local_rank=0,
#         rank=0,
#     )

#     worker.init_device()
#     worker.load_model()

#     # 1. No LoRA Request
#     # 2. 1 LoRA Request
#     # 3. Multiple LoRA Requests
#     # 4. .

def get_lora_requests():
    lora_requests = [
        LoRARequest(str(i + 1), i + 1, "/path/adapter" + str(i + 1))
        for i in range(NUM_LORAS)
    ]
    return lora_requests


def requests_processing_time(llm,
                                   lora_requests: list[LoRARequest]) -> float:

    sampling_params = SamplingParams(n=1,
                                     temperature=0.0,
                                     top_p=1.0,
                                     ignore_eos=True,
                                     max_tokens=1)

    generators = []

    for lora_request in lora_requests:
        lora_int_id = lora_request.lora_int_id
        generator = llm.add_request(
            prompt=TextPrompt(prompt=f"hello {lora_int_id}",
                              multi_modal_data=None),  # type: ignore 
            sampling_params=sampling_params,
            lora_request=lora_request,
            request_id=f"test{lora_int_id}")
        generators.append(generator)

    all_gens = merge_async_iterators(*generators)
    async for i, res in all_gens:
        pass

def fake_load_model(self):
    self.model = MagicMock(spec=RBLNOptimumForCausalLM)
    self.model.kv_block_adapter = KVCacheBlockAdapter(
        vllm_config=get_vllm_config(),
        estimated_kvcache_num_blocks=NUM_BLOCKS + 1,
    )
    self.use_lora = True
    self.valid_lora_ids = list(range(NUM_LORAS + 1))

@pytest.mark.asyncio
async def test_add_lora():
    engine_args = AsyncEngineArgs(
        # FIXME patch is required
        model="llama3.1-8b-ab-sec-b4",
        enable_lora=True,
        max_loras=NUM_LORAS,
        max_lora_rank=8,
        max_model_len=128,
        max_num_batched_tokens=128,
        max_num_seqs=2,
        block_size=BLOCK_SIZE,
    )

    lora_requests = get_lora_requests()
    with patch("vllm_rbln.worker.optimum_model_runner.RBLNOptimumModelRunner.load_model", fake_load_model):
        async with build_async_engine_client_from_engine_args(engine_args, disable_frontend_multiprocessing=True) as llm:
            await requests_processing_time(llm, lora_requests)
        
