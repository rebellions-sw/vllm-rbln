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
from transformers import AutoTokenizer
from vllm import AsyncEngineArgs, AsyncLLMEngine, SamplingParams

INPUT_PROMPT = "Translate English to French: How are you?"


async def generate(
    engine: AsyncLLMEngine,
    tokenizer: AutoTokenizer,
    prompt: str,
    model: str,
    requst_id: int,
    truncate_prompt_tokens: int,
):
    print(f"generate request_id={requst_id}, prompt={prompt}")
    example_input = {
        "temperature": 0.0,
        "request_id": str(requst_id),
    }
    # start the generation
    encoder_prompt_token_ids = tokenizer.encode(
        prompt, truncation=True, max_length=truncate_prompt_tokens)
    results_generator = engine.generate(
        prompt={
            "encoder_prompt": {
                "prompt_token_ids": encoder_prompt_token_ids,
            },
            "decoder_prompt": ""
        },
        sampling_params=SamplingParams(
            temperature=example_input["temperature"],
            ignore_eos=False,
            skip_special_tokens=True,
            stop_token_ids=[tokenizer.eos_token_id],
            truncate_prompt_tokens=truncate_prompt_tokens,
        ),
        request_id=example_input["request_id"],
    )

    # get the results
    final_output = None
    async for request_output in results_generator:
        final_output = request_output
    return final_output


def get_input_prompts(num_prompts=10) -> list[str]:
    return [INPUT_PROMPT] * num_prompts


def show_result(result):
    for i, r in enumerate(result):
        inference_output_text = r.outputs[0].text
        print(inference_output_text)


async def main(
    num_input_prompt: int,
    truncate_prompt_tokens: int,
    model_id: str,
):
    engine_args = AsyncEngineArgs(model=model_id)

    engine = AsyncLLMEngine.from_engine_args(engine_args)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    prompt = get_input_prompts(num_prompts=num_input_prompt)
    futures = []
    for i, p in enumerate(prompt):
        if i == num_input_prompt:
            break

        futures.append(
            asyncio.create_task(
                generate(
                    engine,
                    tokenizer,
                    prompt=p,
                    model=model_id,
                    requst_id=i,
                    truncate_prompt_tokens=truncate_prompt_tokens,
                )))

    result = await asyncio.gather(*futures)

    show_result(result)


def entry_point(
    num_input_prompt: int = 1,
    truncate_prompt_tokens: int = 200,
    model_id: str = "/t5-3b-b4",
):
    asyncio.run(
        main(
            num_input_prompt=num_input_prompt,
            truncate_prompt_tokens=truncate_prompt_tokens,
            model_id=model_id,
        ))


if __name__ == "__main__":
    fire.Fire(entry_point)
