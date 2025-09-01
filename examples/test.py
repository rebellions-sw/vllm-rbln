from vllm import LLM, SamplingParams
from vllm.transformers_utils.config import get_hf_text_config

prompts = [
    #"Hello, my name is",
    #"The president of the United States is",
    #"The financial minister of the United States is",
    #"The future of AI is",
    "The capital of France is",
    #"The capital of UK is",
]
llama_1b_model_id = "meta-llama/Llama-3.2-1B"
llama_8b_model_id = "meta-llama/Meta-Llama-3-8B"
llama4_maverick_model_id = "meta-llama/Llama-4-Maverick-17B-128E"

deepseek_v2_lite_model_id = "deepseek-ai/DeepSeek-V2-Lite"

qwen1_5_moe_model_id = "Qwen/Qwen1.5-MoE-A2.7B"
qwen3_0_6_model_id = "Qwen/Qwen3-0.6B"
qwen3_1_7_model_id = "Qwen/Qwen3-1.7B"
qwen3_4_model_id = "Qwen/Qwen3-4B"
qwen3_8_model_id = "Qwen/Qwen3-8B"
qwen3_14_model_id = "Qwen/Qwen3-14B"
qwen3_32_model_id = "Qwen/Qwen3-32B"
qwen3_30_moe_model_id = "Qwen/Qwen3-30B-A3B"
qwen3_235_moe_model_id = "Qwen/Qwen3-235B-A22B"

hf_overrides_kw = {
    "num_hidden_layers": 2,
}

# update config of multi-modal language model num_hidden_layers
def custom_hf_overrides_kw(hf_config):
    if hasattr(hf_config, "text_config"):
        hf_text_config = get_hf_text_config(hf_config)
        hf_text_config.update(hf_overrides_kw)
    else:
        hf_config.update(hf_overrides_kw)
    return hf_config


#model_id = llama_1b_model_id
#model_id = qwen3_1_7_model_id
model_id = llama4_maverick_model_id

#model_id = llama_8b_model_id
#model_id = qwen1_5_moe_model_id
#model_id = qwen3_30_moe_model_id
#model_id = qwen3_235_moe_model_id
#model_id = deepseek_v2_lite_model_id

import os
## The profile results can be visualized using https://ui.perfetto.dev/
profile_dir = './profile/' + model_id.replace('/', '_')
os.environ['VLLM_TORCH_PROFILER_DIR'] = profile_dir

# Create a sampling params object.
sampling_params = SamplingParams(temperature=0.0)
warmup_sampling_params = SamplingParams(temperature=0.0, max_tokens=2)
llm = LLM(
    model=model_id,
    #hf_overrides=hf_overrides_kw,
    hf_overrides=custom_hf_overrides_kw,
    # max_model_len=40 * 1024,
    max_model_len=8 * 1024,
    block_size=1024,
    enable_chunked_prefill=True,
    max_num_batched_tokens=128,
    max_num_seqs=1,
    trust_remote_code=True,
    tensor_parallel_size=8,
    enable_expert_parallel=True,
)

# 1. warmup -  The first run initializes the compiled models.
# warmup will remove compilation time from profile results.
llm.generate(".", warmup_sampling_params)

# Generate texts from the prompts. The output is a list of RequestOutput objects
# that contain the prompt, generated text, and other information.

# 2. vllm torch profiler will capture model inference time into profile directory
llm.start_profile()
outputs = llm.generate(prompts, sampling_params)
llm.stop_profile()
# Print the outputs.
for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
