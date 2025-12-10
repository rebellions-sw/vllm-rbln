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

import atexit
import csv
import json
import os
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Optional

from vllm_rbln.logger import init_logger

logger = init_logger(__name__)


@dataclass
class StepMetrics:
    """Metrics for a single execution step."""
    latencies: list[float] = field(default_factory=list)
    token_counts: list[int] = field(default_factory=list)
    batch_sizes: list[int] = field(default_factory=list)
    seq_lens_list: list[list[int]] = field(default_factory=list)
    query_lens_list: list[list[int]] = field(default_factory=list)
    timestamps: list[float] = field(default_factory=list)

    def add_measurement(
        self,
        latency: float,
        token_count: int,
        batch_size: Optional[int] = None,
        seq_lens: Optional[list[int]] = None,
        query_lens: Optional[list[int]] = None,
    ):
        """Add a latency and token count measurement with optional details."""
        self.latencies.append(latency)
        self.token_counts.append(token_count)
        self.timestamps.append(time.time())
        if batch_size is not None:
            self.batch_sizes.append(batch_size)
        if seq_lens is not None:
            self.seq_lens_list.append(seq_lens.copy())
        if query_lens is not None:
            self.query_lens_list.append(query_lens.copy())

    def _without_outlier(self, values: list[float | int]) -> list[float | int]:
        """Return values excluding one outlier (max absolute deviation)."""
        if len(values) <= 1:
            return values
        mean = sum(values) / len(values)
        deviations = [abs(v - mean) for v in values]
        max_idx = deviations.index(max(deviations))
        return [v for i, v in enumerate(values) if i != max_idx]

    def _get_percentile(self, values: list[float], percentile: float) -> float:
        """Get percentile value from a list."""
        if not values:
            return 0.0
        sorted_values = sorted(values)
        idx = int(len(sorted_values) * percentile / 100)
        idx = min(idx, len(sorted_values) - 1)
        return sorted_values[idx]

    def _get_std(self, values: list[float]) -> float:
        """Get standard deviation."""
        if len(values) < 2:
            return 0.0
        mean = sum(values) / len(values)
        variance = sum((x - mean) ** 2 for x in values) / len(values)
        return variance ** 0.5

    def get_avg_latency(self, ignore_outlier: bool = True) -> float:
        """Get average latency in milliseconds, optionally ignoring one outlier."""
        values = self._without_outlier(self.latencies) if ignore_outlier else self.latencies
        return sum(values) / len(values) * 1000 if values else 0.0

    def get_min_latency(self) -> float:
        """Get minimum latency in milliseconds."""
        return min(self.latencies) * 1000 if self.latencies else 0.0

    def get_max_latency(self) -> float:
        """Get maximum latency in milliseconds."""
        return max(self.latencies) * 1000 if self.latencies else 0.0

    def get_std_latency(self) -> float:
        """Get standard deviation of latency in milliseconds."""
        return self._get_std(self.latencies) * 1000

    def get_percentile_latency(self, percentile: float) -> float:
        """Get percentile latency in milliseconds."""
        return self._get_percentile(self.latencies, percentile) * 1000

    def get_avg_throughput(self, ignore_outlier: bool = True) -> float:
        """Get average throughput in tokens/second, optionally ignoring one outlier."""
        if not self.latencies or not self.token_counts:
            return 0.0
        latencies = self._without_outlier(self.latencies) if ignore_outlier else self.latencies
        tokens = self._without_outlier(self.token_counts) if ignore_outlier else self.token_counts
        total_time = sum(latencies)
        total_tokens = sum(tokens)
        return total_tokens / total_time if total_time > 0 else 0.0

    def get_per_step_throughputs(self) -> list[float]:
        """Get throughput for each step in tokens/second."""
        throughputs = []
        for lat, tok in zip(self.latencies, self.token_counts):
            if lat > 0:
                throughputs.append(tok / lat)
            else:
                throughputs.append(0.0)
        return throughputs

    def get_call_counts(self) -> int:
        """Get total number of requests processed."""
        return len(self.latencies)

    def get_avg_batch_size(self) -> float:
        """Get average batch size."""
        return sum(self.batch_sizes) / len(self.batch_sizes) if self.batch_sizes else 0.0

    def get_detailed_stats(self) -> dict[str, Any]:
        """Get comprehensive statistics dictionary."""
        latencies_ms = [lat * 1000 for lat in self.latencies]
        throughputs = self.get_per_step_throughputs()

        stats = {
            "call_counts": self.get_call_counts(),
            "total_tokens": sum(self.token_counts),
            "latency": {
                "avg_ms": self.get_avg_latency(ignore_outlier=False),
                "avg_ms_no_outlier": self.get_avg_latency(ignore_outlier=True),
                "min_ms": self.get_min_latency(),
                "max_ms": self.get_max_latency(),
                "std_ms": self.get_std_latency(),
                "p50_ms": self.get_percentile_latency(50),
                "p90_ms": self.get_percentile_latency(90),
                "p95_ms": self.get_percentile_latency(95),
                "p99_ms": self.get_percentile_latency(99),
            },
            "throughput": {
                "avg_tokens_per_sec": self.get_avg_throughput(ignore_outlier=False),
                "avg_tokens_per_sec_no_outlier": self.get_avg_throughput(ignore_outlier=True),
                "min_tokens_per_sec": min(throughputs) if throughputs else 0.0,
                "max_tokens_per_sec": max(throughputs) if throughputs else 0.0,
            },
        }

        if self.batch_sizes:
            stats["batch_size"] = {
                "avg": self.get_avg_batch_size(),
                "min": min(self.batch_sizes),
                "max": max(self.batch_sizes),
            }

        return stats

    def get_raw_data(self) -> list[dict[str, Any]]:
        """Get raw measurement data as list of dictionaries."""
        raw_data = []
        for i in range(len(self.latencies)):
            entry = {
                "step": i + 1,
                "timestamp": self.timestamps[i] if i < len(self.timestamps) else None,
                "latency_ms": self.latencies[i] * 1000,
                "token_count": self.token_counts[i],
                "throughput_tokens_per_sec": (
                    self.token_counts[i] / self.latencies[i]
                    if self.latencies[i] > 0 else 0.0
                ),
            }
            if i < len(self.batch_sizes):
                entry["batch_size"] = self.batch_sizes[i]
            if i < len(self.seq_lens_list):
                entry["seq_lens"] = self.seq_lens_list[i]
            if i < len(self.query_lens_list):
                entry["query_lens"] = self.query_lens_list[i]
            raw_data.append(entry)
        return raw_data


class PerformanceTracker:
    """Tracks performance metrics for prefill and decode steps."""

    def __init__(self, output_dir: Optional[str] = None):
        self.prefill_metrics = StepMetrics()
        self.decode_metrics = StepMetrics()
        self._registered_cleanup = False
        self._start_time = time.time()
        self._output_dir = output_dir or os.environ.get(
            "VLLM_RBLN_METRICS_OUTPUT_DIR", "./rbln_metrics"
        )

    def register_cleanup(self):
        """Register cleanup function to print stats on exit."""
        if not self._registered_cleanup:
            atexit.register(self.print_final_stats)
            self._registered_cleanup = True

    def record_prefill(
        self,
        latency: float,
        token_count: int,
        batch_size: Optional[int] = None,
        seq_lens: Optional[list[int]] = None,
        query_lens: Optional[list[int]] = None,
    ):
        """Record prefill step metrics with optional details."""
        self.prefill_metrics.add_measurement(
            latency, token_count, batch_size, seq_lens, query_lens
        )

    def record_decode(
        self,
        latency: float,
        token_count: int,
        batch_size: Optional[int] = None,
        seq_lens: Optional[list[int]] = None,
        query_lens: Optional[list[int]] = None,
    ):
        """Record decode step metrics with optional details."""
        self.decode_metrics.add_measurement(
            latency, token_count, batch_size, seq_lens, query_lens
        )

    def _print_detailed_metrics(self, name: str, metrics: StepMetrics):
        """Print detailed metrics for a step type."""
        if metrics.get_call_counts() == 0:
            logger.info("%s METRICS: No data recorded", name)
            return

        stats = metrics.get_detailed_stats()

        logger.info("%s METRICS:", name)
        logger.info("  Call Statistics:")
        logger.info("    Total call counts: %d", stats["call_counts"])
        logger.info("    Total tokens processed: %d", stats["total_tokens"])

        if "batch_size" in stats:
            logger.info("  Batch Size Statistics:")
            logger.info("    Average: %.2f", stats["batch_size"]["avg"])
            logger.info("    Min: %d", stats["batch_size"]["min"])
            logger.info("    Max: %d", stats["batch_size"]["max"])

        logger.info("  Latency Statistics (ms):")
        lat = stats["latency"]
        logger.info("    Average: %.3f (without outlier: %.3f)",
                    lat["avg_ms"], lat["avg_ms_no_outlier"])
        logger.info("    Min: %.3f", lat["min_ms"])
        logger.info("    Max: %.3f", lat["max_ms"])
        logger.info("    Std Dev: %.3f", lat["std_ms"])
        logger.info("    P50: %.3f", lat["p50_ms"])
        logger.info("    P90: %.3f", lat["p90_ms"])
        logger.info("    P95: %.3f", lat["p95_ms"])
        logger.info("    P99: %.3f", lat["p99_ms"])

        logger.info("  Throughput Statistics (tokens/sec):")
        thr = stats["throughput"]
        logger.info("    Average: %.2f (without outlier: %.2f)",
                    thr["avg_tokens_per_sec"], thr["avg_tokens_per_sec_no_outlier"])
        logger.info("    Min: %.2f", thr["min_tokens_per_sec"])
        logger.info("    Max: %.2f", thr["max_tokens_per_sec"])

    def _print_raw_data_table(self, name: str, metrics: StepMetrics,
                               max_rows: int = 20):
        """Print raw data as a formatted table."""
        raw_data = metrics.get_raw_data()
        if not raw_data:
            return

        logger.info("")
        logger.info("%s RAW DATA (showing %d of %d entries):",
                    name, min(max_rows, len(raw_data)), len(raw_data))
        logger.info("  %-6s | %-12s | %-10s | %-18s | %-10s",
                    "Step", "Latency(ms)", "Tokens", "Throughput(tok/s)", "BatchSize")
        logger.info("  %s", "-" * 70)

        # Show first max_rows/2 and last max_rows/2 entries if too many
        if len(raw_data) > max_rows:
            half = max_rows // 2
            entries_to_show = raw_data[:half] + [None] + raw_data[-half:]
        else:
            entries_to_show = raw_data

        for entry in entries_to_show:
            if entry is None:
                logger.info("  %s", "... (truncated) ...")
                continue
            batch_size = entry.get("batch_size", "-")
            logger.info("  %-6d | %-12.3f | %-10d | %-18.2f | %-10s",
                        entry["step"],
                        entry["latency_ms"],
                        entry["token_count"],
                        entry["throughput_tokens_per_sec"],
                        batch_size)

    def _save_to_files(self):
        """Save metrics to CSV and JSON files."""
        try:
            os.makedirs(self._output_dir, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

            # Save JSON with full statistics
            json_path = os.path.join(self._output_dir,
                                      f"metrics_{timestamp}.json")
            metrics_data = {
                "timestamp": timestamp,
                "total_runtime_sec": time.time() - self._start_time,
                "prefill": {
                    "statistics": self.prefill_metrics.get_detailed_stats(),
                    "raw_data": self.prefill_metrics.get_raw_data(),
                },
                "decode": {
                    "statistics": self.decode_metrics.get_detailed_stats(),
                    "raw_data": self.decode_metrics.get_raw_data(),
                },
            }
            with open(json_path, "w") as f:
                json.dump(metrics_data, f, indent=2, default=str)
            logger.info("Metrics saved to JSON: %s", json_path)

            # Save CSV for raw data
            for name, metrics in [("prefill", self.prefill_metrics),
                                   ("decode", self.decode_metrics)]:
                raw_data = metrics.get_raw_data()
                if not raw_data:
                    continue

                csv_path = os.path.join(self._output_dir,
                                         f"{name}_raw_{timestamp}.csv")
                fieldnames = ["step", "timestamp", "latency_ms", "token_count",
                              "throughput_tokens_per_sec", "batch_size",
                              "seq_lens", "query_lens"]
                with open(csv_path, "w", newline="") as f:
                    writer = csv.DictWriter(f, fieldnames=fieldnames,
                                            extrasaction="ignore")
                    writer.writeheader()
                    for entry in raw_data:
                        # Convert lists to strings for CSV
                        row = entry.copy()
                        if "seq_lens" in row:
                            row["seq_lens"] = str(row["seq_lens"])
                        if "query_lens" in row:
                            row["query_lens"] = str(row["query_lens"])
                        writer.writerow(row)
                logger.info("Raw data saved to CSV: %s", csv_path)

        except Exception as e:
            logger.warning("Failed to save metrics files: %s", e)

    def print_final_stats(self):
        """Print comprehensive final statistics."""
        total_runtime = time.time() - self._start_time

        logger.info("")
        logger.info("=" * 80)
        logger.info("FINAL PERFORMANCE STATISTICS")
        logger.info("=" * 80)
        logger.info("Total Runtime: %.2f seconds", total_runtime)
        logger.info("")

        # Detailed prefill stats
        self._print_detailed_metrics("PREFILL", self.prefill_metrics)
        self._print_raw_data_table("PREFILL", self.prefill_metrics)

        logger.info("")
        logger.info("-" * 80)
        logger.info("")

        # Detailed decode stats
        self._print_detailed_metrics("DECODE", self.decode_metrics)
        self._print_raw_data_table("DECODE", self.decode_metrics)

        logger.info("")
        logger.info("=" * 80)

        # Summary comparison
        if (self.prefill_metrics.get_call_counts() > 0 and
                self.decode_metrics.get_call_counts() > 0):
            logger.info("SUMMARY COMPARISON:")
            logger.info("  %-15s | %-15s | %-15s", "", "Prefill", "Decode")
            logger.info("  %s", "-" * 50)
            logger.info("  %-15s | %-15d | %-15d", "Call Counts",
                        self.prefill_metrics.get_call_counts(),
                        self.decode_metrics.get_call_counts())
            logger.info("  %-15s | %-15d | %-15d", "Total Tokens",
                        sum(self.prefill_metrics.token_counts),
                        sum(self.decode_metrics.token_counts))
            logger.info("  %-15s | %-15.2f | %-15.2f", "Avg Latency(ms)",
                        self.prefill_metrics.get_avg_latency(),
                        self.decode_metrics.get_avg_latency())
            logger.info("  %-15s | %-15.2f | %-15.2f", "Throughput(t/s)",
                        self.prefill_metrics.get_avg_throughput(),
                        self.decode_metrics.get_avg_throughput())
            logger.info("=" * 80)

        # Save to files
        save_files = os.environ.get("VLLM_RBLN_METRICS_SAVE_FILES", "True").lower() in ("true", "1")
        if save_files:
            self._save_to_files()