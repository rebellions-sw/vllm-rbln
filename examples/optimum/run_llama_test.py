from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

# Make sure the engine configuration
# matches the parameters used during compilation.
model_id = "/home/eunji.lee/nas_data/0713_models/llama3-8b-b2"
max_seq_len = 8192
batch_size = 2

llm = LLM(
    model=model_id,
    device="auto",
    max_num_seqs=batch_size,
    max_num_batched_tokens=max_seq_len,
    max_model_len=max_seq_len,
    block_size=max_seq_len,
)

tokenizer = AutoTokenizer.from_pretrained(model_id)

sampling_params = SamplingParams(
  temperature=0.0,
  skip_special_tokens=True,
  seed=42,
  stop_token_ids=[tokenizer.eos_token_id],
)

conversation = [
    {
        "role": "user",
        "content": "What is the first letter of English alphabets?"
    }
]

chat = tokenizer.apply_chat_template(
  conversation, 
  add_generation_prompt=True,
  tokenize=False
)

outputs = llm.generate(chat, sampling_params)
for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")