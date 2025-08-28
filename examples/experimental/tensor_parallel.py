from vllm import LLM, SamplingParams
import resource
import os
# Set the maximum memory limit (in bytes)
# memory_limit = 80 * 1024 * 1024 * 1024  # 80 GB
# resource.setrlimit(resource.RLIMIT_AS, (memory_limit, memory_limit))
if __name__ == '__main__':
    # Sample prompts.
    prompts = [
        # "Hello, my name is",
        "The president of the United States is",
        # "The capital of France is",
        # "The future of AI is",
    ]
    # Create a sampling params object.
    # sampling_params = SamplingParams(temperature=0.0, max_tokens=2)
    sampling_params = SamplingParams(temperature=0.0)

    hf_overrides = {
        "num_hidden_layers": 16,
    }
    
    # Create an LLM.
    llm = LLM(model="meta-llama/Llama-3.2-1B",
            hf_overrides=hf_overrides,
            max_model_len=128*1024,
            tensor_parallel_size=2,
            block_size=1024,
            # distributed_executor_backend="mp",
            distributed_executor_backend="external_launcher",
            seed=0,)

    # Generate texts from the prompts. The output is a list of RequestOutput objects
    # that contain the prompt, generated text, and other information.
    outputs = llm.generate(prompts, sampling_params)
    # Print the outputs.
    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
        print(output.outputs[0])