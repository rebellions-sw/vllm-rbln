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

import fire
from vllm import LLM, SamplingParams


def get_sampling_params() -> SamplingParams:
    return SamplingParams(
        temperature=0,
        top_p=1.0,
        ignore_eos=True,
        max_tokens=80,
    )


def get_input_prompts(num_input_prompt: int) -> list[str]:
    base_prompts = [
        "Hello, my name is",
        "The president of the United States is",
        "The capital of France is",
        "The future of AI is",
        "In a shocking finding, scientists discovered",
        "The quick brown fox jumps over the lazy dog",
        "To be or not to be, that is",
        "Once upon a time in a land far, far away,",
        "When is the next solar eclipse?",
        "The recipe for a perfect chocolate cake includes",
    ]
    
    # Use modular arithmetic to cycle through prompts if num_input_prompt > len(base_prompts)
    return [base_prompts[i % len(base_prompts)] for i in range(num_input_prompt)]

def main(
    batch_size: int,
    max_seq_len: int,
    kvcache_block_size: int,
    num_input_prompt: int,
    model_id: str,
):
    llm = LLM(model=model_id,
              max_num_seqs=batch_size,
              max_num_batched_tokens=max_seq_len,
              max_model_len=max_seq_len,
              block_size=kvcache_block_size)
    prompts = get_input_prompts(num_input_prompt)
    sampling_params = get_sampling_params()
    outputs = llm.generate(prompts, sampling_params)

    # Print the outputs.
    print("-" * 50)
    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        print(f"Prompt: {prompt!r}\nGenerated text: {generated_text!r}")
        print("-" * 50)


def entry_point(
    batch_size: int = 4,
    max_seq_len: int = 131072,
    kvcache_block_size: int = 16_384,
    num_input_prompt: int = 4,
    model_id: str = "./Llama-3.1-8B-Instruct-b4",
):
    main(
        batch_size=batch_size,
        max_seq_len=max_seq_len,
        kvcache_block_size=kvcache_block_size,
        num_input_prompt=num_input_prompt,
        model_id=model_id,
    )


if __name__ == "__main__":
    fire.Fire(entry_point)