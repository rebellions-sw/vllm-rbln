import time

from vllm import LLM, SamplingParams
from vllm.assets.audio import AudioAsset
from datasets import load_dataset
from transformers import AutoTokenizer
# Create a Whisper encoder/decoder model instance
model_id = "/home/eunji.lee/nas_data/0704_models/whisper-base-b4-wo-token-timestamps"
batch_size = 4
llm = LLM(
    model=model_id,
    device="auto",
    block_size=448,
    max_num_batched_tokens=448,
    max_model_len=448,
    max_num_seqs=batch_size,
    limit_mm_per_prompt={"audio": 4},
)
tokenizer = AutoTokenizer.from_pretrained(model_id)

ds = load_dataset("distil-whisper/librispeech_asr-noise", "test-pub-noise", split="40")

prompts = [
    {
        "prompt": "<|startoftranscript|>",
        "multi_modal_data": {
            # "audio": AudioAsset("mary_had_lamb").audio_and_sample_rate,
            "audio": (ds[i]["audio"]["array"], ds[i]["audio"]["sampling_rate"])
        },
    } for i in range(10)
]

# Create a sampling params object.
sampling_params = SamplingParams(
    temperature=0,
    max_tokens=500,
    ignore_eos=False,
    skip_special_tokens=True,
    stop_token_ids=[tokenizer.eos_token_id],
)

start = time.time()

# Generate output tokens from the prompts. The output is a list of
# RequestOutput objects that contain the prompt, generated
# text, and other information.
outputs = llm.generate(prompts, sampling_params)

# Print the outputs.
for output in outputs:
    prompt = output.prompt
    encoder_prompt = output.encoder_prompt
    generated_text = output.outputs[0].text
    print(f"Encoder prompt: {encoder_prompt!r}, "
          f"Decoder prompt: {prompt!r}, "
          f"Generated text: {generated_text!r}")

duration = time.time() - start

print("Duration:", duration)
print("RPS:", len(prompts) / duration)