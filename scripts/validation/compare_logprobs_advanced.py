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

import argparse
import os
import urllib.request
from multiprocessing import get_context

import torch
from transformers import AutoTokenizer

MODEL_NAME = "meta-llama/Llama-3.2-1B"
PREFILL_CHUNK_SIZE = 128
VOCAB_SIZE = 128256
EPSILON = 1e-1 * 5


def get_wiki_prompt():
    wiki_txt_url = "https://raw.githubusercontent.com/huggingface/optimum-neuron/refs/heads/main/benchmark/text-generation/performance/wiki.txt"
    with urllib.request.urlopen(wiki_txt_url) as resp:
        source_data = resp.read().decode("utf-8")
    return source_data


def generate_llm_args(device):
    llm_args = {
        "model": "meta-llama/Llama-3.2-1B",
        "max_model_len": 40 * 1024,
        "enable_chunked_prefill": True,
        "max_num_seqs": 1,
        "max_logprobs": VOCAB_SIZE,
    }
    if device == "cpu":
        llm_args["block_size"] = 128
    elif device == "rbln":
        llm_args["block_size"] = 128  # 1024 is not working for long prompt
        llm_args["max_num_batched_tokens"] = PREFILL_CHUNK_SIZE
    return llm_args


def generate_prompts(prompt_length, batch_size) -> list[str]:
    wiki_prompt = get_wiki_prompt()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokens = tokenizer(wiki_prompt, return_tensors="pt").input_ids[0]
    assert len(tokens) > prompt_length * batch_size
    prompts = []
    for i in range(batch_size):
        prompt = tokenizer.decode(tokens[i * prompt_length:(i + 1) *
                                         prompt_length])
        prompts.append(prompt)
    return prompts


def run_llm(llm, prompts, sampling_params, q):
    outputs = llm.generate(prompts, sampling_params=sampling_params)
    q.put(outputs)


def _worker(device, prompts, q, args):
    llm_args = generate_llm_args(device)
    if device == "cpu":
        os.environ["VLLM_PLUGINS"] = "cpu"
        os.environ["VLLM_USE_V1"] = "0"
    elif device == "rbln":
        os.environ.pop("VLLM_PLUGINS", None)
        os.environ["RBLN_KERNEL_MODE"] = "triton"
        os.environ["VLLM_USE_V1"] = "0"
        os.environ["USE_VLLM_MODEL"] = "1"
        os.environ["VLLM_DISABLE_COMPILE_CACHE"] = "1"
        # 1 means disable using compile cache
    else:
        raise ValueError(f"Unknown device: {device}")

    from vllm import LLM, SamplingParams
    sampling_params = SamplingParams(
        temperature=0.0,
        top_p=1.0,
        ignore_eos=True,
        max_tokens=args.max_tokens,
        logprobs=VOCAB_SIZE,
    )
    llm = LLM(**llm_args)
    run_llm(llm, prompts, sampling_params, q)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-l", "--prompt_length", type=int, default=128)
    parser.add_argument("-m", "--max_tokens", type=int, default=1)
    parser.add_argument("-b", "--batch_size", type=int, default=1)
    args = parser.parse_args()

    torch.manual_seed(42)

    prompts = generate_prompts(args.prompt_length, args.batch_size)

    ctx = get_context("spawn")
    q = ctx.Queue()
    p1 = ctx.Process(target=_worker, args=("cpu", prompts, q, args))
    p1.start()
    cpu_outputs = q.get()
    p1.join()

    p2 = ctx.Process(target=_worker, args=("rbln", prompts, q, args))
    p2.start()
    rbln_outputs = q.get()
    p2.join()

    if p1.exitcode != 0 or p2.exitcode != 0:
        raise SystemExit("One of the processes worked incorrectly.")

    for cpu_output, rbln_output in zip(cpu_outputs, rbln_outputs):
        print("=========" * 10)
        cpu_logprobs = cpu_output.outputs[0].logprobs
        rbln_logprobs = rbln_output.outputs[0].logprobs
        num_outlier = 0
        for cpu_lp_token_id, cpu_lp_score in cpu_logprobs[0].items():
            cpu_logprob = cpu_lp_score.logprob
            if cpu_lp_token_id not in rbln_logprobs[0]:
                continue
            rbln_logprob = rbln_logprobs[0].get(cpu_lp_token_id).logprob
            if abs(cpu_logprob - rbln_logprob) >= EPSILON:
                num_outlier += 1
        print(f"Number of outliers: {num_outlier}")
        print(f"Prompt: {cpu_output.prompt}")
        print(f"Generated text  (CPU): {cpu_output.outputs[0].text}")
        print(f"Generated text (RBLN): {rbln_output.outputs[0].text}")
