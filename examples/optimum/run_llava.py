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
from transformers import AutoProcessor, AutoTokenizer
from vllm import AsyncEngineArgs, AsyncLLMEngine, SamplingParams


def generate_prompts(batch_size: int, model_id: str):
    dataset = load_dataset("lmms-lab/llava-bench-in-the-wild",
                           split="train").shuffle(seed=42)
    processor = AutoProcessor.from_pretrained(model_id, padding_side="left")
    messages = [[
        {
            "role":
            "system",
            "content": [{
                "type":
                "text",
                "text":
                "You are a helpful assistant."
                "Answer the each question based on the image.",
            }],
        },
        {
            "role":
            "user",
            "content": [
                {
                    "type": "image"
                },
                {
                    "type": "text",
                    "text": dataset[i]["question"]
                },
            ],
        },
    ] for i in range(batch_size)]
    images = [[dataset[i]["image"]] for i in range(batch_size)]

    texts = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=False,
    )

    return [{
        "prompt": text,
        "multi_modal_data": {
            "image": image
        }
    } for text, image in zip(texts, images)]


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
    kvcache_partition_len: int,
    num_input_prompt: int,
    model_id: str,
):
    engine_args = AsyncEngineArgs(model=model_id,
                                  device="auto",
                                  max_num_seqs=batch_size,
                                  max_num_batched_tokens=max_seq_len,
                                  max_model_len=max_seq_len,
                                  block_size=kvcache_partition_len)

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
    max_seq_len: int = 32768,
    kvcache_partition_len: int = 16384,
    num_input_prompt: int = 10,
    model_id: str = "/llava-v1.6-mistral-7b-hf-32k-b4-kv16k",
):
    loop = asyncio.get_event_loop()
    loop.run_until_complete(
        main(
            batch_size=batch_size,
            max_seq_len=max_seq_len,
            kvcache_partition_len=kvcache_partition_len,
            num_input_prompt=num_input_prompt,
            model_id=model_id,
        ))

if __name__ == "__main__":
    fire.Fire(entry_point)
