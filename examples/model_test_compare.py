#!/usr/bin/env python3
import os
import argparse
import json
from multiprocessing import get_context
from functools import partial

# ========= Tunables =========
EPSILON = 1e-1 * 5
# ============================

DEFAULT_PROMPTS = [
    "Hello, my name is",
    "The president of the United States is",
    "The capital of France is",
    "The future of AI is",
]

# ---------- CLI ----------
parser = argparse.ArgumentParser(description="CPU vs RBLN parity runner (separate processes; clean envs).")
parser.add_argument('--model', type=str, default="llama3.2-1b",
                    choices=[
                        "llama3.2-1b", "llama3-8b", "qwen3-1.7b",
                        "qwen1.5-moe-15b", "qwen3-moe-30b", "qwen3-moe-235b",
                        "deepseek-v2", "llama4-maverick",
                    ])
parser.add_argument('--tp', type=int, default=1)
parser.add_argument('--pp', type=int, default=1)
parser.add_argument('--dp', type=int, default=1)
parser.add_argument('--ep', action='store_true')
parser.add_argument('--profile', action='store_true')
parser.add_argument('--max-model-len', type=int, default=40 * 1024)
parser.add_argument('--block-size-cpu', type=int, default=128)
parser.add_argument('--block-size-rbln', type=int, default=8 * 1024)
parser.add_argument('--max-batched', type=int, default=128)
parser.add_argument('--prompts', type=str, nargs='*', default=None)
parser.add_argument('--trust-remote-code', action='store_true')
parser.add_argument('--use-cache', action='store_true', help="Use cached CPU results if available")
parser.add_argument('--num-hidden-layers', type=int, default=None, help="Override model hidden layer count")
parser.add_argument('--max-tokens', type=int, default=256, help="Number of tokens to generate per prompt")
parser.add_argument('--logprobs', type=int, default=1024,
                    help="Per-token logprobs: 0=off; N=top-N; -1=full vocab")
parser.add_argument('--max-logprobs-cap', type=int, default=128256,
                    help='Engine-wide cap for logprobs; must be >= requested logprobs')

args = parser.parse_args()

# ---------- Model map ----------
MODELS = {
    "llama3.2-1b": ("meta-llama/Llama-3.2-1B", False),
    "llama3-8b": ("meta-llama/Meta-Llama-3-8B", False),
    "qwen3-1.7b": ("Qwen/Qwen3-1.7B", False),
    "qwen1.5-moe-15b": ("Qwen/Qwen1.5-MoE-A2.7B", True),
    "qwen3-moe-30b": ("Qwen/Qwen3-30B-A3B", True),
    "qwen3-moe-235b": ("Qwen/Qwen3-235B-A22B", True),
    "deepseek-v2": ("deepseek-ai/DeepSeek-V2-Lite", True),
    "llama4-maverick": ("meta-llama/Llama-4-Maverick-17B-128E", True),
}

assert args.model in MODELS, "Invalid model name"
model_id, should_ep = MODELS[args.model]
if should_ep:
    assert args.ep, f"{args.model} requires --ep"
else:
    assert not args.ep, f"{args.model} should not use --ep"

prompts = args.prompts if args.prompts else DEFAULT_PROMPTS
print(f"Model = {model_id}, EP={args.ep}, TP={args.tp}, PP={args.pp}, DP={args.dp}, "
      f"MaxTokens={args.max_tokens}, Logprobs={args.logprobs}")

# ---------- Top-level HF override (PICKLABLE) ----------
def hf_override_num_layers(hf_config, num_hidden_layers: int):
    """Top-level override so it's picklable under spawn."""
    from vllm.transformers_utils.config import get_hf_text_config
    if hasattr(hf_config, "text_config"):
        txt = get_hf_text_config(hf_config)
        txt.update({"num_hidden_layers": num_hidden_layers})
    else:
        hf_config.update({"num_hidden_layers": num_hidden_layers})
    return hf_config

# ---------- Helpers ----------
def generate_llm_args(device: str):
    llm_args = {
        "model": model_id,
        "max_model_len": args.max_model_len,
        "enable_chunked_prefill": True,
        "max_num_seqs": 1,
        "max_num_batched_tokens": args.max_batched,
        "trust_remote_code": args.trust_remote_code,
        "max_logprobs": args.max_logprobs_cap,
    }

    if device == "rbln":
        # Keep distributed & expert parallel settings only for RBLN
        llm_args.update({
            "tensor_parallel_size": args.tp,
            "pipeline_parallel_size": args.pp,
            "data_parallel_size": args.dp,
            "enable_expert_parallel": args.ep,
        })
    else:
        # CPU: force single process; no EP
        llm_args.update({
            "tensor_parallel_size": 1,
            "pipeline_parallel_size": 1,
            "data_parallel_size": 1,
            "enable_expert_parallel": False,
        })

    if args.num_hidden_layers is not None:
        # Use partial of a top-level function to stay picklable
        llm_args["hf_overrides"] = partial(hf_override_num_layers,
                                           num_hidden_layers=args.num_hidden_layers)

    llm_args["block_size"] = args.block_size_cpu if device == "cpu" else args.block_size_rbln
    return llm_args


def set_env_for_device(device: str):
    """Configure environment variables for each backend (must run in the target process)."""
    for k in [
        "VLLM_PLUGINS", "RBLN_KERNEL_MODE", "USE_VLLM_MODEL", "VLLM_USE_V1",
        "VLLM_DISABLE_COMPILE_CACHE", "VLLM_TORCH_PROFILER_DIR",
    ]:
        os.environ.pop(k, None)

    if device == "cpu":
        os.environ["VLLM_PLUGINS"] = "cpu"
        os.environ["VLLM_USE_V1"] = "0"
    elif device == "rbln":
        # Important: don't set VLLM_PLUGINS for RBLN
        os.environ["RBLN_KERNEL_MODE"] = "triton"
        os.environ["VLLM_USE_V1"] = "0"
        os.environ["USE_VLLM_MODEL"] = "1"
        os.environ["VLLM_DISABLE_COMPILE_CACHE"] = "1"
    else:
        raise ValueError(f"Unknown device: {device}")

    if args.profile:
        profile_dir = f'./profile/{device}_{model_id.replace("/", "_")}'
        os.makedirs(profile_dir, exist_ok=True)
        os.environ['VLLM_TORCH_PROFILER_DIR'] = profile_dir


def cache_path():
    os.makedirs("./cache", exist_ok=True)
    key = f"{args.model.replace('/', '_')}"
    if args.num_hidden_layers:
        key += f"_L{args.num_hidden_layers}"
    key += f"_T{args.max_tokens}"
    if args.logprobs == -1:
        key += "_LPvocab"
    elif args.logprobs > 0:
        key += f"_LP{args.logprobs}"
    return f"./cache/cpu_results_{key}.json"


def _extract_logprobs_dict(request_output) -> dict:
    """Return {token_id(str): logprob(float)} for position 0 if available."""
    lp = getattr(request_output.outputs[0], "logprobs", None)
    if isinstance(lp, list) and lp and isinstance(lp[0], dict):
        try:
            return {str(k): v.logprob for k, v in lp[0].items()}
        except Exception:
            return {}
    return {}


def _pack_outputs(outputs):
    """Convert vLLM RequestOutput[] -> serializable list of dicts."""
    packed = []
    for o in outputs:
        packed.append({
            "prompt": o.prompt,
            "text": o.outputs[0].text,
            "logprobs": _extract_logprobs_dict(o),
        })
    return packed


def save_cpu_results(packed_outputs):
    path = cache_path()
    with open(path, "w") as f:
        json.dump(packed_outputs, f, indent=2)
    print(f"✅ Cached CPU results to {path}")


def load_cpu_results():
    path = cache_path()
    if not os.path.exists(path):
        return None
    with open(path, "r") as f:
        data = json.load(f)
    print(f"💾 Loaded cached CPU results from {path}")
    return data


def compare_and_print(cpu_packed, rbln_packed):
    for cpu_output, rbln_output in zip(cpu_packed, rbln_packed):
        print("=========" * 10)
        prompt = cpu_output.get("prompt", "<unknown>")
        cpu_text = cpu_output.get("text", "")
        rbln_text = rbln_output.get("text", "")

        cpu_logprobs = cpu_output.get("logprobs", {}) or {}
        rbln_logprobs = rbln_output.get("logprobs", {}) or {}

        num_outlier = 0
        for token_id_str, cpu_lp in cpu_logprobs.items():
            try:
                token_id = int(token_id_str)
            except ValueError:
                continue
            r_lp = rbln_logprobs.get(str(token_id))
            if r_lp is None and isinstance(next(iter(rbln_logprobs.values()), None), float):
                # If RBLN map keys are ints-as-ints; normalize to str keys
                r_lp = rbln_logprobs.get(token_id)
            if r_lp is None:
                continue
            rbln_lp = r_lp if isinstance(r_lp, float) else r_lp  # already float
            if abs(cpu_lp - rbln_lp) >= EPSILON:
                num_outlier += 1

        print(f"Prompt: {prompt}")
        print(f"Generated text  (CPU): {cpu_text}")
        print(f"Generated text (RBLN): {rbln_text}")
        print(f"Number of outliers: {num_outlier}")


# ----------------- Multiprocess worker -----------------
def _worker(device: str, q, prompts_local, logprobs_flag, max_tokens_local):
    """Child process: sets env, builds LLM, runs warmup + generate, returns PACKED outputs (dicts)."""
    set_env_for_device(device)
    # Import vLLM only after env is set
    from vllm import LLM, SamplingParams

    llm_args = generate_llm_args(device)
    llm = LLM(**llm_args)

    # Warmup (compile)
    llm.generate(".", SamplingParams(temperature=0.0, max_tokens=2))

    # Resolve logprobs count
    if logprobs_flag == 0:
        lp_count = None
    elif logprobs_flag == -1:
        # Full vocab if possible
        lp_count = None
        try:
            tok = llm.get_tokenizer()
            lp_count = getattr(tok, "vocab_size", None) or len(tok)
        except Exception:
            lp_count = None
    else:
        lp_count = logprobs_flag

    sampling_params = SamplingParams(
        temperature=0.0,
        ignore_eos=True,
        max_tokens=max_tokens_local,
        logprobs=lp_count,
    )

    print(f"[{device}] VLLM_PLUGINS = {os.environ.get('VLLM_PLUGINS', '<unset>')}")
    outputs = llm.generate(prompts_local, sampling_params)

    # PACK to avoid pickling heavy vLLM objects / closures
    q.put(_pack_outputs(outputs))


def main():
    # 1) CPU phase: use cache or compute in fresh process
    if args.use_cache:
        cpu_packed = load_cpu_results()
        if cpu_packed is None:
            ctx = get_context("spawn")
            q = ctx.Queue()
            print("\n[main] Launching CPU worker...")
            p1 = ctx.Process(target=_worker, args=("cpu", q, prompts, args.logprobs, args.max_tokens))
            p1.start()
            cpu_packed = q.get()
            p1.join()
            if p1.exitcode != 0:
                raise SystemExit("CPU process failed.")
            save_cpu_results(cpu_packed)
    else:
        ctx = get_context("spawn")
        q = ctx.Queue()
        print("\n[main] Launching CPU worker...")
        p1 = ctx.Process(target=_worker, args=("cpu", q, prompts, args.logprobs, args.max_tokens))
        p1.start()
        cpu_packed = q.get()
        p1.join()
        if p1.exitcode != 0:
            raise SystemExit("CPU process failed.")
        save_cpu_results(cpu_packed)

    # 2) RBLN phase: always fresh process (no cache)
    ctx = get_context("spawn")
    q = ctx.Queue()
    print("\n[main] Launching RBLN worker...")
    p2 = ctx.Process(target=_worker, args=("rbln", q, prompts, args.logprobs, args.max_tokens))
    p2.start()
    rbln_packed = q.get()
    p2.join()
    if p2.exitcode != 0:
        raise SystemExit("RBLN process failed.")

    # 3) Compare packed dicts from both sides
    compare_and_print(cpu_packed, rbln_packed)


if __name__ == "__main__":
    main()