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
from dataclasses import dataclass, field

from vllm_rbln.logger import init_logger

logger = init_logger(__name__)


@dataclass
class StepMetrics:
    """Metrics for a single execution step."""
    latencies: list[float] = field(default_factory=list)
    token_counts: list[int] = field(default_factory=list)

    def add_measurement(self, latency: float, token_count: int):
        """Add a latency and token count measurement."""
        self.latencies.append(latency)
        self.token_counts.append(token_count)

    def _without_outlier_f(self, values: list[float]) -> list[float]:
        """Return values excluding one outlier (max absolute deviation)."""
        if len(values) <= 1:
            return values
        mean = sum(values) / len(values)
        deviations = [abs(v - mean) for v in values]
        max_idx = deviations.index(max(deviations))
        return [v for i, v in enumerate(values) if i != max_idx]

    def _without_outlier_i(self, values: list[int]) -> list[int]:
        """Return values excluding one outlier (max absolute deviation)."""
        if len(values) <= 1:
            return values
        mean = sum(values) / len(values)
        deviations = [abs(v - mean) for v in values]
        max_idx = deviations.index(max(deviations))
        return [v for i, v in enumerate(values) if i != max_idx]

    def get_avg_latency(self, ignore_outlier: bool = True) -> float:
        """Get average latency in milliseconds,
        optionally ignoring one outlier."""
        values = self._without_outlier_f(
            self.latencies) if ignore_outlier else self.latencies
        return sum(values) / len(values) * 1000 if values else 0.0

    def get_avg_throughput(self, ignore_outlier: bool = True) -> float:
        """Get average throughput in tokens/second,
        optionally ignoring one outlier."""
        if not self.latencies or not self.token_counts:
            return 0.0
        latencies = self._without_outlier_f(
            self.latencies) if ignore_outlier else self.latencies
        tokens = self._without_outlier_i(
            self.token_counts) if ignore_outlier else self.token_counts
        total_time = sum(latencies)
        total_tokens = sum(tokens)
        return total_tokens / total_time if total_time > 0 else 0.0

    def get_call_counts(self) -> int:
        """Get total number of requests processed."""
        return len(self.latencies)


class PerformanceTracker:
    """Tracks performance metrics for prefill and decode steps."""

    def __init__(self):
        self.prefill_metrics = StepMetrics()
        self.decode_metrics = StepMetrics()
        self._registered_cleanup = False

    def register_cleanup(self):
        """Register cleanup function to print stats on exit."""
        if not self._registered_cleanup:
            atexit.register(self.print_final_stats)
            self._registered_cleanup = True

    def record_prefill(self, latency: float, token_count: int):
        """Record prefill step metrics."""
        self.prefill_metrics.add_measurement(latency, token_count)

    def record_decode(self, latency: float, token_count: int):
        """Record decode step metrics."""
        self.decode_metrics.add_measurement(latency, token_count)

    def print_final_stats(self):
        logger.info("=" * 80)
        logger.info("FINAL PERFORMANCE STATISTICS")
        logger.info("=" * 80)

        # Prefill stats
        if self.prefill_metrics.get_call_counts() > 0:
            logger.info("PREFILL METRICS:")
            logger.info("  Total call counts: %d",
                        self.prefill_metrics.get_call_counts())
            logger.info("  Total tokens processed: %d",
                        sum(self.prefill_metrics.token_counts))
            logger.info("  Average latency: %.2f ms",
                        self.prefill_metrics.get_avg_latency())
            logger.info("  Average throughput: %.2f tokens/sec",
                        self.prefill_metrics.get_avg_throughput())
        else:
            logger.info("PREFILL METRICS: No data recorded")

        logger.info("-" * 40)

        # Decode stats
        if self.decode_metrics.get_call_counts() > 0:
            logger.info("DECODE METRICS:")
            logger.info("  Total call counts: %d",
                        self.decode_metrics.get_call_counts())
            logger.info("  Total tokens processed: %d",
                        sum(self.decode_metrics.token_counts))
            logger.info("  Average latency: %.2f ms",
                        self.decode_metrics.get_avg_latency())
            logger.info("  Average throughput: %.2f tokens/sec",
                        self.decode_metrics.get_avg_throughput())
        else:
            logger.info("DECODE METRICS: No data recorded")

        logger.info("=" * 80)
