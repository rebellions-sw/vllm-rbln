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

import fire
from datasets import load_dataset
from transformers import AutoTokenizer
from vllm import AsyncEngineArgs, AsyncLLMEngine, SamplingParams


def generate_prompts(batch_size: int, model_id: str):
    dataset = load_dataset("lmms-lab/llava-bench-in-the-wild",
                           split="train").shuffle(seed=42)

    prompts = []
    for i in range(batch_size):
        image = dataset[i]["image"]
        question = dataset[i]["question"]

        # Use simple QA template because BLIP2 don't have default chat template.
        text_prompt = (f"Question: {question}\n"
                       "Answer:")

        prompts.append({
            "prompt": text_prompt,
            "multi_modal_data": {
                "image": [image]
            }
        })

    return prompts


async def generate(engine: AsyncLLMEngine, tokenizer, request_id, request):
    results_generator = engine.generate(
        request,
        SamplingParams(temperature=0,
                       ignore_eos=False,
                       skip_special_tokens=True,
                       stop_token_ids=[tokenizer.eos_token_id],
                       max_tokens=200),
        str(request_id),
    )

    final_output = None
    async for request_output in results_generator:
        final_output = request_output
    return final_output


async def main(
    batch_size: int,
    max_seq_len: int,
    num_input_prompt: int,
    model_id: str,
):
    engine_args = AsyncEngineArgs(model=model_id,
                                  device="auto",
                                  max_num_seqs=batch_size,
                                  max_num_batched_tokens=max_seq_len,
                                  max_model_len=max_seq_len,
                                  block_size=max_seq_len)

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
    batch_size: int = 4,
    max_seq_len: int = 2048,
    num_input_prompt: int = 10,
    model_id: str = "/blip2-opt-2.7b-2k-b4",
):
    loop = asyncio.get_event_loop()
    loop.run_until_complete(
        main(
            batch_size=batch_size,
            max_seq_len=max_seq_len,
            num_input_prompt=num_input_prompt,
            model_id=model_id,
        ))


if __name__ == "__main__":
    fire.Fire(entry_point)
