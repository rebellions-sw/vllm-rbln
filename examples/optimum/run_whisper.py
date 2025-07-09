import time

from vllm import LLM, SamplingParams
from vllm.assets.audio import AudioAsset
from datasets import load_dataset
# Create a Whisper encoder/decoder model instance
model_id = "whisper-tiny-b4-wo-token-timestamps"
llm = LLM(
    model=model_id,
    device="auto",
    block_size=448,
    max_num_batched_tokens=448,
    max_model_len=448,
    max_num_seqs=4,
    limit_mm_per_prompt={"audio": 1},
    kv_cache_dtype="fp8",
)

ds = load_dataset("distil-whisper/librispeech_long", "clean", split="validation")
audio_array = ds[0]["audio"]["array"]
sampling_rate = ds[0]["audio"]["sampling_rate"]
prompts = [
    {
        "prompt": "<|startoftranscript|>",
        "multi_modal_data": {
            # "audio": AudioAsset("mary_had_lamb").audio_and_sample_rate,
            "audio": (audio_array, sampling_rate)
        },
    },
]

# Create a sampling params object.
sampling_params = SamplingParams(
    temperature=0,
    top_p=1.0,
    max_tokens=400,
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