# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import argparse
import atexit
import dataclasses
import json
import math
import os
import shutil
import time
from types import TracebackType
from typing import Any, Optional, Union

from vllm import LLM, EngineArgs

from vllm_rbln.rbln_envs import RBLN_USE_VLLM_MODEL


def get_llm_instance(engine_args: EngineArgs) -> LLM:
    if RBLN_USE_VLLM_MODEL:  # torch.compile path
        return LLM(**dataclasses.asdict(engine_args))

    # optimum-rbln path
    from optimum.rbln import RBLNAutoModelForCausalLM

    attn_impl = os.getenv("OPTIMUM_RBLN_ATTN_IMPL", "flash_attn")

    # NOTE: Cache usage and rebuild policy (needs verification)
    # We currently rebuild Optimum-RBLN artifacts on every benchmark run and
    # remove the cache at exit.
    # It is not yet confirmed which EngineArgs changes require a rebuild.
    # Do NOT enable cache reuse until this is verified.
    # If/when enabling, compute a cache key from the confirmed build-affecting
    # params (e.g., a hash) and a separate cache directory per key.
    cache_path = os.path.join(os.path.dirname(__file__), ".optimum-rbln-cache")
    model = RBLNAutoModelForCausalLM.from_pretrained(
        engine_args.model,
        export=True,
        rbln_create_runtimes=False,
        rbln_batch_size=engine_args.max_num_seqs,
        rbln_tensor_parallel_size=engine_args.tensor_parallel_size,
        rbln_max_seq_len=engine_args.max_model_len,
        rbln_kvcache_partition_len=engine_args.block_size
        if attn_impl == "flash_attn" else None,
        rbln_kvcache_block_size=engine_args.block_size,
        rbln_prefill_chunk_size=engine_args.max_num_batched_tokens
        if engine_args.enable_chunked_prefill else None,
        rbln_attn_impl=attn_impl,
    )
    model.save_pretrained(cache_path)

    def _shutdown_handler():
        done = {"x": False}

        def _run_once():
            if done["x"]:
                return
            done["x"] = True

            if os.path.exists(cache_path):
                shutil.rmtree(cache_path)

        atexit.register(_run_once)

    _shutdown_handler()
    engine_args.model = cache_path
    return LLM(**dataclasses.asdict(engine_args))


def convert_to_pytorch_benchmark_format(args: argparse.Namespace,
                                        metrics: dict[str, list],
                                        extra_info: dict[str, Any]) -> list:
    """
    Save the benchmark results in the format used by PyTorch OSS benchmark with
    on metric per record
    https://github.com/pytorch/pytorch/wiki/How-to-integrate-with-PyTorch-OSS-benchmark-database
    """
    records = []
    if not os.environ.get("SAVE_TO_PYTORCH_BENCHMARK_FORMAT", False):
        return records

    for name, benchmark_values in metrics.items():
        record = {
            "benchmark": {
                "name": "vLLM benchmark",
                "extra_info": {
                    "args": vars(args),
                },
            },
            "model": {
                "name": args.model,
            },
            "metric": {
                "name": name,
                "benchmark_values": benchmark_values,
                "extra_info": extra_info,
            },
        }

        tp = record["benchmark"]["extra_info"]["args"].get(
            "tensor_parallel_size")
        # Save tensor_parallel_size parameter if it's part of the metadata
        if not tp and "tensor_parallel_size" in extra_info:
            record["benchmark"]["extra_info"]["args"][
                "tensor_parallel_size"] = (extra_info["tensor_parallel_size"])

        records.append(record)

    return records


class InfEncoder(json.JSONEncoder):

    def clear_inf(self, o: Any):
        if isinstance(o, dict):
            return {k: self.clear_inf(v) for k, v in o.items()}
        elif isinstance(o, list):
            return [self.clear_inf(v) for v in o]
        elif isinstance(o, float) and math.isinf(o):
            return "inf"
        return o

    def iterencode(self, o: Any, *args, **kwargs) -> Any:
        return super().iterencode(self.clear_inf(o), *args, **kwargs)


def write_to_json(filename: str, records: list) -> None:
    with open(filename, "w") as f:
        json.dump(
            records,
            f,
            cls=InfEncoder,
            default=lambda o:
            f"<{type(o).__name__} object is not JSON serializable>",
        )


# Collect time and generate time metrics
#
# Example Usage:
#   collector = TimeCollector(TimeCollector.US)
#   for _ in range(total_iteration):
#      with collector:
#          ...
#   collector.dump_avg_max()
class TimeCollector:
    NS: int = 1
    US: int = NS * 1000
    MS: int = US * 1000
    S: int = MS * 1000

    def __init__(self, scale: int) -> None:
        self.cnt: int = 0
        self._sum: int = 0
        self._max: Optional[int] = None
        self.scale = scale
        self.start_time: int = time.monotonic_ns()

    def collect(self, v: int) -> None:
        self.cnt += 1
        self._sum += v
        if self._max is None:
            self._max = v
        else:
            self._max = max(self._max, v)

    def avg(self) -> Union[float, str]:
        return (self._sum * 1.0 / self.cnt /
                self.scale if self.cnt > 0 else "N/A")

    def max(self) -> Union[float, str]:
        return self._max / self.scale if self._max else "N/A"

    def dump_avg_max(self) -> list[Union[float, str]]:
        return [self.avg(), self.max()]

    def __enter__(self) -> None:
        self.start_time = time.monotonic_ns()

    def __exit__(
        self,
        exc_type: Optional[type[BaseException]],
        exc_value: Optional[BaseException],
        exc_traceback: Optional[TracebackType],
    ) -> None:
        self.collect(time.monotonic_ns() - self.start_time)
