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
import math

import fire
from transformers import AutoTokenizer
from vllm import AsyncEngineArgs, AsyncLLMEngine, SamplingParams
from vllm.inputs.data import TokensPrompt

SUFFIX = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"


def format_instruction(instruction, query, doc):
    text = [{
        "role":
        "system",
        "content": ("Judge whether the Document meets the requirements "
                    "based on the Query and the Instruct provided. "
                    "Note that the answer can only be \"yes\" or \"no\".")
    }, {
        "role":
        "user",
        "content":
        f"<Instruct>: {instruction}\n\n<Query>: {query}\n\n<Document>: {doc}"
    }]
    return text


def process_inputs(pairs, instruction, max_length, suffix_tokens, tokenizer):
    messages = [
        format_instruction(instruction, query, doc) for query, doc in pairs
    ]
    messages = tokenizer.apply_chat_template(messages,
                                             tokenize=True,
                                             add_generation_prompt=False,
                                             enable_thinking=False)
    messages = [ele[:max_length] + suffix_tokens for ele in messages]
    messages = [TokensPrompt(prompt_token_ids=ele) for ele in messages]
    return messages


def get_input_prompts(model_id, max_length, suffix_tokens,
                      tokenizer) -> list[str]:
    task = ('Given a web search query, '
            'retrieve relevant passages that answer the query')
    queries = [
        "What is the capital of China?",
        "Explain gravity",
    ]
    documents = [
        "The capital of China is Beijing.",
        ("Gravity is a force that attracts two bodies towards each other. "
         "It gives weight to physical objects and "
         "is responsible for the movement of planets around the sun.")
    ]

    pairs = list(zip(queries, documents))
    inputs = process_inputs(pairs, task, max_length - len(suffix_tokens),
                            suffix_tokens, tokenizer)

    return inputs


async def generate(engine: AsyncLLMEngine, prompt_tokens: list[int],
                   model: str, request_id: int, true_token: int,
                   false_token: int):
    print(f"generate request_id={request_id}, prompt_tokens={prompt_tokens}")
    example_input = {
        "stream": True,
        "temperature": 0.0,
        "request_id": str(request_id),
    }

    sampling_params = SamplingParams(
        temperature=0,
        max_tokens=1,
        logprobs=20,
        allowed_token_ids=[true_token, false_token],
    )

    results_generator = engine.generate(
        prompt_tokens,
        sampling_params,
        example_input["request_id"],
    )

    # get the results
    final_output = None
    async for request_output in results_generator:
        final_output = request_output
    return final_output


def compute_logits(outputs, true_token, false_token):
    scores = []
    for i in range(len(outputs)):
        final_logits = outputs[i].outputs[0].logprobs[-1]
        if true_token not in final_logits:
            true_logit = -10
        else:
            true_logit = final_logits[true_token].logprob
        if false_token not in final_logits:
            false_logit = -10
        else:
            false_logit = final_logits[false_token].logprob
        true_score = math.exp(true_logit)
        false_score = math.exp(false_logit)
        score = true_score / (true_score + false_score)
        scores.append(score)
    return scores


async def main(
    max_seq_len: int,
    num_input_prompt: int,
    model_id: str,
):
    engine_args = AsyncEngineArgs(model=model_id)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.padding_side = "left"
    tokenizer.pad_token = tokenizer.eos_token
    suffix_tokens = tokenizer.encode(SUFFIX, add_special_tokens=False)

    true_token = tokenizer("yes", add_special_tokens=False).input_ids[0]
    false_token = tokenizer("no", add_special_tokens=False).input_ids[0]

    engine = AsyncLLMEngine.from_engine_args(engine_args)
    prompt_tokens_list = get_input_prompts(model_id, max_seq_len,
                                           suffix_tokens, tokenizer)
    futures = []
    for i, p in enumerate(prompt_tokens_list):
        if i == num_input_prompt:
            break

        futures.append(
            asyncio.create_task(
                generate(engine,
                         prompt_tokens=p,
                         model=model_id,
                         request_id=i,
                         true_token=true_token,
                         false_token=false_token)))

    result = await asyncio.gather(*futures)
    score = compute_logits(result, true_token, false_token)
    print(f"scores: {score}")


def entry_point(
    max_seq_len: int = 32768,
    num_input_prompt: int = 2,
    model_id: str = "/qwen3-0.6b-b1",
):
    asyncio.run(
        main(
            max_seq_len=max_seq_len,
            num_input_prompt=num_input_prompt,
            model_id=model_id,
        ))


if __name__ == "__main__":
    fire.Fire(entry_point)
