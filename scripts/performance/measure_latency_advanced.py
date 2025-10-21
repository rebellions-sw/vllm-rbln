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

# Reference - https://github.com/vllm-project/vllm/blob/v0.9.1/benchmarks/benchmark_throughput.py
import argparse
import os
import time
import urllib.request
from typing import TYPE_CHECKING, Any

import torch
from transformers import AutoTokenizer

if TYPE_CHECKING:
    from vllm import SamplingParams
    from vllm.outputs import RequestOutput

MODEL_NAME = "meta-llama/Llama-3.2-1B"
PREFILL_CHUNK_SIZE = 128


def get_wiki_prompt():
    wiki_txt_url = "https://raw.githubusercontent.com/huggingface/optimum-neuron/refs/heads/main/benchmark/text-generation/performance/wiki.txt"
    with urllib.request.urlopen(wiki_txt_url) as resp:
        source_data = resp.read().decode("utf-8")
    return source_data


def generate_llm_args(batch_size: int):
    return {
        "model": "meta-llama/Llama-3.2-1B",
        "max_model_len": 40 * 1024,
        "enable_chunked_prefill": True,
        "max_num_seqs": batch_size,
        "block_size": 1024,
        "max_num_batched_tokens": PREFILL_CHUNK_SIZE,
    }


def generate_prompts(prompt_length: int, batch_size: int) -> list[str]:
    wiki_prompt = get_wiki_prompt()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokens = tokenizer(wiki_prompt, return_tensors="pt").input_ids[0]
    assert len(tokens) > prompt_length * batch_size
    prompts = []
    # Leave 1 token for special token(bos) in the vllm
    real_prompt_length = prompt_length - 1
    for i in range(batch_size):
        start_pos = i * real_prompt_length
        end_pos = (i + 1) * real_prompt_length
        prompt = tokenizer.decode(tokens[start_pos:end_pos])
        prompts.append(prompt)
    return prompts


def run_llm(
        llm, prompts: list[str], sampling_params: "SamplingParams"
) -> tuple[float, list["RequestOutput"]]:
    start = time.perf_counter()
    outputs = llm.generate(prompts, sampling_params=sampling_params)
    end = time.perf_counter()
    elapsed_time = end - start
    return elapsed_time, outputs


def _worker(prompts: list[str], args: Any):
    llm_args = generate_llm_args(args.batch_size)
    os.environ["VLLM_RBLN_METRICS"] = "1"
    os.environ.pop("VLLM_PLUGINS", None)
    os.environ["RBLN_KERNEL_MODE"] = "triton"
    os.environ["VLLM_USE_V1"] = "0"
    os.environ["USE_VLLM_MODEL"] = "1"
    os.environ["VLLM_DISABLE_COMPILE_CACHE"] = "0"
    # 1 means disable using compile cache
    from vllm import LLM, SamplingParams
    sampling_params = SamplingParams(
        temperature=0.0,
        top_p=1.0,
        ignore_eos=True,
        max_tokens=args.max_tokens,
    )
    total_elapsed_time = 0.0
    # FIXME: In rbln, re-initializing LLM
    # in each iteration triggers runtime error:
    # (Runtime) code=203 INIT_ALREADY_CREATED:
    # A runtime has already been created for that compiled model
    # (Context failed to be created, compile_id=0).
    # Try creating a runtime on a different NPU(s), or use an existing runtime.
    llm = LLM(**llm_args)
    for _ in range(args.num_iter):
        elapsed_time, outputs = run_llm(llm, prompts, sampling_params)
        total_elapsed_time += elapsed_time
    return total_elapsed_time


def calculate_avg_throughput_and_latency(elapsed_time: float, batch_size: int,
                                         max_tokens: int,
                                         num_iter: int) -> tuple[float, float]:
    avg_throughput = (batch_size * max_tokens * num_iter) / elapsed_time
    avg_latency = elapsed_time / num_iter
    return avg_throughput, avg_latency


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-l", "--prompt_length", type=int, default=128)
    parser.add_argument("-m", "--max_tokens", type=int, default=1)
    parser.add_argument("-b", "--batch_size", type=int, default=1)
    parser.add_argument("-n", "--num_iter", type=int, default=1)
    args = parser.parse_args()

    torch.manual_seed(42)

    prompts = generate_prompts(args.prompt_length, args.batch_size)
    elapsed_time = _worker(prompts, args)
