# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

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

import os
import random
import tempfile
from unittest.mock import patch

from vllm.config import (
    CacheConfig,
    DeviceConfig,
    ModelConfig,
    ParallelConfig,
    SchedulerConfig,
    VllmConfig,
    set_current_vllm_config,
)
from vllm.config.load import LoadConfig
from vllm.config.lora import LoRAConfig
from vllm.lora.models import LoRAMapping
from vllm.lora.request import LoRARequest

from vllm_rbln.v1.worker.rbln_worker import RBLNWorker as Worker

NUM_LORAS = 16


@patch.dict(os.environ, {"RANK": "0"})
def test_worker_apply_lora(sql_lora_files):
    def set_active_loras(worker: Worker, lora_requests: list[LoRARequest]):
        lora_mapping = LoRAMapping([], [])

        worker.model_runner.lora_manager.set_active_adapters(
            lora_requests, lora_mapping
        )

    vllm_config = VllmConfig(
        model_config=ModelConfig(
            "meta-llama/Llama-2-7b-hf",
            seed=0,
            dtype="bfloat16",
            task="generate",
            disable_cascade_attn=True,
        ),
        load_config=LoadConfig(
            download_dir=None,
            load_format="dummy",
        ),
        parallel_config=ParallelConfig(
            pipeline_parallel_size=1,
            tensor_parallel_size=1,
            data_parallel_size=1,
        ),
        scheduler_config=SchedulerConfig("generate", 32, 32, 32),
        device_config=DeviceConfig("cpu"),
        cache_config=CacheConfig(
            block_size=16,
            swap_space=0,
            cache_dtype="auto",
        ),
        lora_config=LoRAConfig(
            max_lora_rank=8, max_cpu_loras=NUM_LORAS, max_loras=NUM_LORAS
        ),
    )
    worker = Worker(
        vllm_config=vllm_config,
        local_rank=0,
        rank=0,
        distributed_init_method=f"file://{tempfile.mkstemp()[1]}",
    )

    with set_current_vllm_config(vllm_config):
        worker.init_device()
    worker.load_model()

    set_active_loras(worker, [])
    assert worker.list_loras() == set()

    lora_requests = [
        LoRARequest(str(i + 1), i + 1, sql_lora_files) for i in range(NUM_LORAS)
    ]

    set_active_loras(worker, lora_requests)
    assert worker.list_loras() == {
        lora_request.lora_int_id for lora_request in lora_requests
    }

    for i in range(NUM_LORAS):
        random.seed(i)
        iter_lora_requests = random.choices(
            lora_requests, k=random.randint(1, NUM_LORAS)
        )
        random.shuffle(iter_lora_requests)
        iter_lora_requests = iter_lora_requests[: -random.randint(0, NUM_LORAS)]
        set_active_loras(worker, lora_requests)
        assert worker.list_loras().issuperset(
            {lora_request.lora_int_id for lora_request in iter_lora_requests}
        )
