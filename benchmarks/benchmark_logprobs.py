# from vllm import LLM, SamplingParams
import os
from multiprocessing import get_context

VOCAB_SIZE = 128256
EPSILON = 1e-1 * 5
llm_args = {
    "model": "meta-llama/Llama-3.2-1B",
    "max_model_len": 40 * 1024,
    "block_size": 1024,
    "enable_chunked_prefill": True,
    "max_num_batched_tokens": 128,
    "max_num_seqs": 1,
    "max_logprobs": VOCAB_SIZE,
}

prompts = [
    "Hello, my name is",
    "The president of the United States is",
    "The capital of France is",
    "The future of AI is",
]

def run_llm(llm, sampling_params, q):
    outputs = llm.generate(prompts, sampling_params)
    q.put(outputs)

def _worker(device, q, llm_args):
    if device == "cpu":
        os.environ["VLLM_PLUGINS"] = "cpu"
        os.environ["VLLM_USE_V1"] = "0"
    elif device == "rbln":
        os.environ.pop("VLLM_PLUGINS", None)
        os.environ["RBLN_KERNEL_MODE"] = "triton"
        os.environ["VLLM_USE_V1"] = "0"
        os.environ["USE_VLLM_MODEL"] = "1"
        os.environ["VLLM_DISABLE_COMPILE_CACHE"] = "0"
    else:
        raise ValueError(f"Unknown device: {device}")

    from vllm import LLM, SamplingParams
    sampling_params = SamplingParams(temperature=0.0, logprobs=VOCAB_SIZE, ignore_eos=True, max_tokens=2)
    llm = LLM(**llm_args)
    run_llm(llm, sampling_params, q)


if __name__ == "__main__":
    ctx = get_context("spawn")
    q = ctx.Queue()
    p1 = ctx.Process(target=_worker, args=("cpu", q, llm_args))
    p1.start()
    cpu_outputs = q.get()
    p1.join()

    p2 = ctx.Process(target=_worker, args=("rbln", q, llm_args))
    p2.start()
    rbln_outputs = q.get()
    p2.join()

    if p1.exitcode != 0 or p2.exitcode != 0:
        raise SystemExit("one of workers failed")

    for cpu_output, rbln_output in zip(cpu_outputs, rbln_outputs):
        print("=========" * 10)
        cpu_logprobs = cpu_output.outputs[0].logprobs
        rbln_logprobs = rbln_output.outputs[0].logprobs
        num_outlier = 0
        for cpu_lp_token_id, cpu_lp_score in cpu_logprobs[0].items():
            cpu_logprob = cpu_lp_score.logprob
            if cpu_lp_token_id not in rbln_logprobs[0]:
                continue
            rbln_logprob = rbln_logprobs[0].get(cpu_lp_token_id).logprob
            # print(f"Token ID: {cpu_lp_token_id:6d}, CPU Logprob: {cpu_logprob:.3f}, RBLN Logprob: {rbln_logprob:.3f}")
            # assert abs(cpu_logprob - rbln_logprob) < EPSILON, f"Logprobs differ: {cpu_logprob} vs {rbln_logprob}"
            if abs(cpu_logprob - rbln_logprob) >= EPSILON:
                num_outlier += 1
        print(f"Number of outliers: {num_outlier}")
