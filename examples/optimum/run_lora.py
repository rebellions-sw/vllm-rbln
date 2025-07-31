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
import json

import fire
from simphile import jaccard_similarity
from transformers import AutoTokenizer
from vllm import AsyncEngineArgs, AsyncLLMEngine, SamplingParams
from datasets import load_dataset

async def generate(engine: AsyncLLMEngine, prompt: str, model: str,
                   request_id: int, max_tokens: int):
    print(f"generate request_id={request_id}, prompt={prompt}")
    example_input = {
        "stream": True,
        "temperature": 0.0,
        "request_id": str(request_id),
    }

    # start the generation
    conversation = [{"role": "user", "content": f"Summarize the paper:\n{prompt}"}]
    tokenizer = AutoTokenizer.from_pretrained(model)
    chat = tokenizer.apply_chat_template(
        conversation,
        add_generation_prompt=True,
        tokenize=False,
    )

    results_generator = engine.generate(
        chat,
        SamplingParams(temperature=example_input["temperature"],
                       ignore_eos=False,
                       skip_special_tokens=True,
                       stop_token_ids=[tokenizer.eos_token_id],
                       max_tokens=max_tokens),
        example_input["request_id"],
    )

    # get the results
    final_output = None
    async for request_output in results_generator:
        final_output = request_output
    return final_output


def get_input_prompts() -> list[str]:
    dataset = load_dataset("common-pile/arxiv_papers")["train"]
    paper_text = dataset[0]["text"]
    paper_text = paper_text[:4096]
    return paper_text


async def main(
    batch_size: int,
    max_seq_len: int,
    kvcache_block_size: int,
    num_input_prompt: int,
    model_id: str,
):
    engine_args = AsyncEngineArgs(model=model_id,
                                  device="auto",
                                  max_num_seqs=batch_size,
                                  max_num_batched_tokens=max_seq_len,
                                  max_model_len=max_seq_len,
                                  block_size=kvcache_block_size)

    engine = AsyncLLMEngine.from_engine_args(engine_args)
    prompt = get_input_prompts()
    futures = []
    for i, p in enumerate(prompt):
        if i == num_input_prompt:
            break

        futures.append(
            asyncio.create_task(
                generate(engine,
                         prompt=p,
                         model=model_id,
                         request_id=i,
                         max_tokens=max_seq_len)))

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
    batch_size: int = 1,
    max_seq_len: int = 8192,
    kvcache_block_size: int = 4096,
    num_input_prompt: int = 1,
    model_id: str = "/llama3.1-8b-b1-lora",
):
    loop = asyncio.get_event_loop()
    loop.run_until_complete(
        main(
            batch_size=batch_size,
            max_seq_len=max_seq_len,
            num_input_prompt=num_input_prompt,
            model_id=model_id,
        ))
    loop.close()


if __name__ == "__main__":
    fire.Fire(entry_point)