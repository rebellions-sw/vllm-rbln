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
from simphile import jaccard_similarity
from transformers import AutoTokenizer
from vllm import AsyncEngineArgs, AsyncLLMEngine, SamplingParams

INPUT_PROMPT = "UN Chief Says There Is No <mask> in Syria"
GOLDEN_PROMPT = "UN Chief Says There Is No Security in Syria"


async def generate(engine: AsyncLLMEngine,
                   prompt: str,
                   model: str,
                   request_id=0):
    print(f"generate request_id={request_id}, prompt={prompt}")
    example_input = {
        "stream": True,
        "temperature": 0.0,
        "request_id": str(request_id),
    }
    # start the generation
    tokenizer = AutoTokenizer.from_pretrained(model)

    results_generator = engine.generate(
        prompt,
        SamplingParams(
            temperature=example_input["temperature"],
            ignore_eos=False,
            skip_special_tokens=True,
            stop_token_ids=[tokenizer.eos_token_id],
        ),
        example_input["request_id"],
    )

    # get the results
    final_output = None
    async for request_output in results_generator:
        final_output = request_output
    return final_output


def get_input_prompts(num_prompts=10) -> list[str]:
    return [INPUT_PROMPT] * num_prompts


def compare(result):
    total_score = 0
    for i, r in enumerate(result):
        inference_output_text = r.outputs[0].text
        print(inference_output_text)

        similarity = jaccard_similarity(GOLDEN_PROMPT, inference_output_text)
        print(f"Similarity score : {similarity}")
        total_score += similarity

    total_avg = total_score / len(result)
    return total_avg


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
    prompt = get_input_prompts(num_prompts=num_input_prompt)
    futures = []
    for i, p in enumerate(prompt):
        if i == num_input_prompt:
            break

        futures.append(
            asyncio.create_task(
                generate(engine, prompt=p, model=model_id, request_id=i)))

    result = await asyncio.gather(*futures)

    score = compare(result)
    if score < 0.97:
        print(f"score is lower than threshold({score})")
        exit(1)


def entry_point(
    batch_size: int = 2,
    max_seq_len: int = 512,
    num_input_prompt: int = 10,
    model_id: str = "/rbln_bart-small_batch2",
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
