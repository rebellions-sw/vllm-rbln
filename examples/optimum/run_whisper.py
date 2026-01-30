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
import asyncio
import os

import fire
from datasets import load_dataset
from transformers import AutoTokenizer
from vllm import AsyncEngineArgs, AsyncLLMEngine, SamplingParams


def generate_prompts(batch_size: int, model_id: str):
    dataset = load_dataset("distil-whisper/librispeech_asr-noise",
                           "test-pub-noise",
                           streaming=True,
                           split="40")
    dataset = dataset.take(batch_size)
    messages = []
    for item in dataset:
        messages.append({
            "prompt": "<|startoftranscript|>",
            "multi_modal_data": {
                "audio":
                (item["audio"]["array"], item["audio"]["sampling_rate"])
            },
        })

    return messages


async def generate(engine: AsyncLLMEngine, tokenizer, request_id, request):
    results_generator = engine.generate(
        request,
        SamplingParams(temperature=0,
                       ignore_eos=False,
                       skip_special_tokens=True,
                       stop_token_ids=[tokenizer.eos_token_id],
                       max_tokens=448),
        str(request_id),
    )

    final_output = None
    async for request_output in results_generator:
        final_output = request_output
    return final_output


async def main(
    num_input_prompt: int,
    model_id: str,
):
    engine_args = AsyncEngineArgs(model=model_id,
                                  limit_mm_per_prompt={"audio": 1})

    engine = AsyncLLMEngine.from_engine_args(engine_args)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    inputs = generate_prompts(num_input_prompt, model_id)

    futures = []
    for request_id, request in enumerate(inputs):
        futures.append(
            asyncio.create_task(
                generate(engine, tokenizer, request_id, request)))

    results = await asyncio.gather(*futures)

    for i, result in enumerate(results):
        output = result.outputs[0].text
        print(
            f"===================== Output {i} ==============================")
        print(output)
        print(
            "===============================================================\n"
        )


def entry_point(
    num_input_prompt: int = 1,
    model_id: str = "/whisper-base-b4-wo-token-timestamps",
):
    asyncio.run(main(
        num_input_prompt=num_input_prompt,
        model_id=model_id,
    ))


if __name__ == "__main__":
    # NOTE To avoid multiprocessing issues
    # `VLLM_WORKER_MULTIPROC_METHOD` must be set to "spawn".
    # for both V0 and V1.
    # https://github.com/vllm-project/vllm/issues/26581
    os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
    fire.Fire(entry_point)
