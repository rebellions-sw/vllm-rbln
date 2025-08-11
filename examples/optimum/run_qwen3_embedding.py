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
import torch
from vllm import AsyncEngineArgs, AsyncLLMEngine, PoolingParams


def get_input_prompts() -> list[str]:

    def get_detailed_instruct(task_description: str, query: str) -> str:
        return f'Instruct: {task_description}\nQuery:{query}'

    # Each query must come with a one-sentence instruction
    # that describes the task
    task = ('Given a web search query, '
            'retrieve relevant passages that answer the query')

    queries = [
        get_detailed_instruct(task, 'What is the capital of China?'),
        get_detailed_instruct(task, 'Explain gravity')
    ]
    documents = [
        "The capital of China is Beijing.",
        ("Gravity is a force that attracts two bodies towards each other. "
         "It gives weight to physical objects and "
         "is responsible for the movement of planets around the sun.")
    ]

    inputs_texts = queries + documents
    return inputs_texts


async def embed(engine: AsyncLLMEngine, prompt: str, model: str,
                requst_id: int):
    print(f"embed request_id={requst_id}, prompt={prompt}")
    pooling_params = PoolingParams()
    results_generator = engine.encode(
        prompt,
        pooling_params,
        str(requst_id),
    )

    # get the results
    final_output = None
    async for request_output in results_generator:
        final_output = request_output
    return final_output


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
                                  block_size=kvcache_block_size,
                                  task="embed")

    engine = AsyncLLMEngine.from_engine_args(engine_args)
    prompt_list = get_input_prompts()
    if len(prompt_list) > 2 * num_input_prompt:
        raise RuntimeError("The len(QUERIES) and len(DOCUMENTS) ",
                           "should be equal with 2 * `num_input_prompt`.")
    futures = []
    for i, p in enumerate(prompt_list):
        if i == num_input_prompt * 2:
            break
        futures.append(
            asyncio.create_task(
                embed(
                    engine,
                    prompt=p,
                    model=model_id,
                    requst_id=i,
                )))

    outputs = await asyncio.gather(*futures)

    embeddings = torch.stack([o.outputs.data for o in outputs])
    scores = (embeddings[:num_input_prompt] @ embeddings[num_input_prompt:].T)

    print(f"scores: {scores.tolist()}")


def entry_point(
    batch_size: int = 1,
    max_seq_len: int = 32768,
    kvcache_block_size: int = 32768,
    num_input_prompt: int = 2,
    model_id: str = "/qwen3-0.6b-b1-embedding",
):
    loop = asyncio.get_event_loop()
    loop.run_until_complete(
        main(
            batch_size=batch_size,
            max_seq_len=max_seq_len,
            kvcache_block_size=kvcache_block_size,
            num_input_prompt=num_input_prompt,
            model_id=model_id,
        ))
    loop.close()


if __name__ == "__main__":
    fire.Fire(entry_point)
