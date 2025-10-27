import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--model', type=str, default="llama3.2-1b", help="model name")
parser.add_argument('--tp', type=int, default=1, help="vLLM tensor_parallel_size")
parser.add_argument('--pp', type=int, default=1, help="vLLM pipeline_parallel_size")
parser.add_argument('--dp', type=int, default=1, help="vLLM data_parallel_size")
parser.add_argument('--ep', action='store_true', help="vLLM enable_expert_parallel")
args = parser.parse_args()

# dense model
llama_1b_model_id = "meta-llama/Llama-3.2-1B"
llama_8b_model_id = "meta-llama/Meta-Llama-3-8B"
qwen3_1_7_model_id = "Qwen/Qwen3-1.7B"

# MoE model
qwen1_5_moe_model_id = "Qwen/Qwen1.5-MoE-A2.7B"
qwen3_30_moe_model_id = "Qwen/Qwen3-30B-A3B"
qwen3_235_moe_model_id = "Qwen/Qwen3-235B-A22B"

# VLLM_MLA_DISABLE=1 DO NOT use MLA attention backend for deepseek
deepseek_v2_lite_model_id = "deepseek-ai/DeepSeek-V2-Lite"
llama4_maverick_model_id = "meta-llama/Llama-4-Maverick-17B-128E"

tensor_parallel_size = args.tp
pipeline_parallel_size = args.pp
data_parallel_size = args.dp
enable_expert_parallel = args.ep

assert data_parallel_size == 1

if args.model == "llama3.2-1b":
    model_id = llama_1b_model_id
    assert not enable_expert_parallel
elif args.model == "llama3-8b":
    model_id = llama_8b_model_id
    assert not enable_expert_parallel
elif args.model == "qwen3-1.7b":
    model_id = qwen3_1_7_model_id
    assert not enable_expert_parallel
elif args.model == "qwen1.5-moe-15b":
    model_id = qwen1_5_moe_model_id
    assert enable_expert_parallel
elif args.model == "qwen3-moe-30b":
    model_id = qwen3_30_moe_model_id
    assert enable_expert_parallel
elif args.model == "qwen3-moe-235b":
    model_id = qwen3_235_moe_model_id
    assert enable_expert_parallel
elif args.model == "deepseek-v2":
    model_id = deepseek_v2_lite_model_id
    assert enable_expert_parallel
elif args.model == "llama4-maverick":
    model_id = llama4_maverick_model_id
    assert enable_expert_parallel
else:
    assert False, "invalid model name"

print(f"model = {model_id}")
print(f"tensor_parallel_size = {args.tp}")
print(f"pipeline_parallel_size = {args.pp}")
print(f"data_parallel_size = {args.dp}")
print(f"enable_expert_parallel = {args.ep}")

from vllm import LLM, SamplingParams
from vllm.transformers_utils.config import get_hf_text_config

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

prompts = [
    # "Hello, my name is",
    "The president of the United States is",
    "The financial minister of the United States is",
    "The future of AI is",
    # "The capital of France is",
    #"The capital of UK is",
    # " a b c d e f g"
]



from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-30B-A3B-Thinking-2507")
# prompts = ["""You are an advanced reasoning assistant built for precise and structured thinking. Your task is to analyse user input carefully, decompose it into logical steps, identify assumptions, and provide a clear final answer. Use step-by-step reasoning when needed. Present your reasoning in numbered or bulleted form if helpful, then summarise your conclusion clearly. If the user’s request is ambiguous, ask a clarifying question instead of guessing. Use code blocks or tables when presenting structured information. At the end ask the user whether they’d like follow-up help or more detail.""" + " no pad" * 9]
# prompts = ["You are a world-class knowledge assistant focused on clear and accurate responses. Carefully read the user’s request, ask clarifying questions if needed, then reason step by step. Use bullet points, numbered steps or headings to organise your answer when helpful. Provide examples, code snippets or diagrams if they improve understanding. At conclusion, summarise your key points in one concise paragraph, and offer to dive deeper if the user wishes. Begin by acknowledging the request. Begin by acknowledging the request. Begin by acknowledging the request. Begin by acknowledging the request. Begin by acknowledging the request. Begin by acknowledging the request. Begin by acknowledging the request. go"]

# tokens = tokenizer(prompts[0])["input_ids"]
# print(len(tokens))
# exit()
# import pdb; pdb.set_trace()


import os
## The profile results can be visualized using https://ui.perfetto.dev/
profile_dir = './profile/' + model_id.replace('/', '_')
os.environ['VLLM_TORCH_PROFILER_DIR'] = profile_dir

max_token = 3
batch_size = 1
n_iter = 4
# Create a sampling params object.
sampling_params = SamplingParams(temperature=0.0, max_tokens=max_token, logprobs=5)
warmup_sampling_params = SamplingParams(temperature=0.0, max_tokens=2)
llm = LLM(
    model=model_id,
    hf_overrides=hf_overrides_kw,
    #hf_overrides=custom_hf_overrides_kw,
    max_model_len=40 * 1024,
    # max_model_len=4 * 1024,
    block_size=8192,
    enable_chunked_prefill=True,
    max_num_batched_tokens=128,
    max_num_seqs=batch_size,
    trust_remote_code=True,
    tensor_parallel_size=tensor_parallel_size,
    pipeline_parallel_size=pipeline_parallel_size,
    data_parallel_size=data_parallel_size,
    enable_expert_parallel=enable_expert_parallel,
)

# 1. warmup -  The first run initializes the compiled models.
# warmup will remove compilation time from profile results.
# llm.generate(".", warmup_sampling_params)

# Generate texts from the prompts. The output is a list of RequestOutput objects
# that contain the prompt, generated text, and other information.

# 2. vllm torch profiler will capture model inference time into profile directory
# llm.start_profile()
all_outputs = []
for i in range(n_iter):
    outputs0 = llm.generate(prompts, sampling_params)
    all_outputs.append(outputs0)
# llm.stop_profile()
    
for outputs in all_outputs:
    print(len(outputs[0].prompt_token_ids))
    for j in range(batch_size):
        print('batch : ', j)
        for i in range(max_token):
            print(outputs[j].outputs[0].logprobs[i])
# # Print the outputs.
# for output in outputs:
#     prompt = output.prompt
#     generated_text = output.outputs[0].text
#     print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
