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
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Optional, TypeVar

from vllm_rbln.logger import init_logger

logger = init_logger(__name__)

T = TypeVar('T', int, float)


def _remove_outlier(values: list[T]) -> list[T]:
    """Return values excluding one outlier (max absolute deviation)."""
    if len(values) <= 1:
        return values
    mean = sum(values) / len(values)
    deviations = [abs(v - mean) for v in values]
    max_idx = deviations.index(max(deviations))
    return [v for i, v in enumerate(values) if i != max_idx]


@dataclass
class BaseStepMetrics:
    """Base class for step metrics."""
    latencies: list[float] = field(default_factory=list)

    def get_call_counts(self) -> int:
        """Get total number of calls processed."""
        return len(self.latencies)

    def get_avg_latency(self, ignore_outlier: bool = True) -> float:
        """Get average latency in milliseconds,
        optionally ignoring one outlier."""
        values = _remove_outlier(
            self.latencies) if ignore_outlier else self.latencies
        return sum(values) / len(values) * 1000 if values else 0.0


@dataclass
class LatencyOnlyMetrics(BaseStepMetrics):
    """Metrics that only track latency (for samplers)."""

    def add_measurement(self, latency: float):
        """Add a latency measurement."""
        self.latencies.append(latency)


@dataclass
class ThroughputMetrics(BaseStepMetrics):
    """Metrics that track both latency and throughput (for models)."""
    token_counts: list[int] = field(default_factory=list)

    def add_measurement(self, latency: float, token_count: int):
        """Add a latency and token count measurement."""
        self.latencies.append(latency)
        self.token_counts.append(token_count)

    def get_total_tokens(self) -> int:
        """Get total number of tokens processed."""
        return sum(self.token_counts)

    def get_avg_throughput(self, ignore_outlier: bool = True) -> float:
        """Get average throughput in tokens/second,
        optionally ignoring one outlier."""
        if not self.latencies or not self.token_counts:
            return 0.0
        latencies = _remove_outlier(
            self.latencies) if ignore_outlier else self.latencies
        tokens = _remove_outlier(
            self.token_counts) if ignore_outlier else self.token_counts
        total_time = sum(latencies)
        total_tokens = sum(tokens)
        return total_tokens / total_time if total_time > 0 else 0.0


class PrefillMetricsByRequestID:
    """Metrics for prefill step tracked by request ID."""

    def __init__(self, metrics_class: type[BaseStepMetrics]):
        """Initialize with the appropriate metrics class."""
        self.metrics_class = metrics_class
        self.metrics: dict[str, BaseStepMetrics] = defaultdict(metrics_class)

    def add_measurement(self,
                        request_id: str,
                        latency: float,
                        token_count: Optional[int] = None):
        """Add a measurement for a specific request."""
        metric = self.metrics[request_id]
        if isinstance(metric, ThroughputMetrics) and token_count is not None:
            metric.add_measurement(latency, token_count)
        elif isinstance(metric, LatencyOnlyMetrics):
            metric.add_measurement(latency)

    def get_avg_latency_per_request(self) -> dict[str, float]:
        """Get average latency per request."""
        return {
            request_id: metric.get_avg_latency()
            for request_id, metric in self.metrics.items()
        }

    def get_num_request_ids(self) -> int:
        """Get total number of request IDs processed."""
        return len(self.metrics)


class BasePerformanceTracker(ABC):
    """Base class for performance trackers with unified API."""

    def __init__(self):
        self._registered_cleanup = False

    def register_cleanup(self):
        """Register cleanup function to print stats on exit."""
        if not self._registered_cleanup:
            atexit.register(self.print_final_stats)
            self._registered_cleanup = True

    @staticmethod
    def _is_dummy_request(request_ids: Optional[list[str]]) -> bool:
        """Check if the request is a dummy request."""
        return bool(request_ids
                    and request_ids[0].startswith("dummy_request_"))

    @abstractmethod
    def record_prefill(self,
                       latency: float,
                       token_count: Optional[int] = None,
                       request_ids: Optional[list[str]] = None):
        """Record prefill step metrics."""
        pass

    @abstractmethod
    def record_decode(self,
                      latency: float,
                      token_count: Optional[int] = None,
                      request_ids: Optional[list[str]] = None):
        """Record decode step metrics."""
        pass

    @abstractmethod
    def print_final_stats(self):
        """Print final statistics."""
        pass


class ModelPerformanceTracker(BasePerformanceTracker):
    """Tracks performance metrics with throughput calculation for models."""

    def __init__(self):
        super().__init__()
        self.prefill_metrics = ThroughputMetrics()
        self.decode_metrics = ThroughputMetrics()
        self.prefill_metrics_by_request_id = PrefillMetricsByRequestID(
            ThroughputMetrics)

    def record_prefill(self,
                       latency: float,
                       token_count: Optional[int] = None,
                       request_ids: Optional[list[str]] = None):
        """Record prefill step metrics."""
        if self._is_dummy_request(request_ids):
            return

        if token_count is None:
            raise ValueError(
                "token_count is required for ModelPerformanceTracker")

        request_id = None
        if request_ids is not None:
            assert len(request_ids) == 1, (
                f"Expected exactly one request_id during prefill, "
                f"got {len(request_ids)}: {request_ids}")
            request_id = request_ids[0]

        self.prefill_metrics.add_measurement(latency, token_count)
        if request_id:
            self.prefill_metrics_by_request_id.add_measurement(
                request_id, latency, token_count)

    def record_decode(self,
                      latency: float,
                      token_count: Optional[int] = None,
                      request_ids: Optional[list[str]] = None):
        """Record decode step metrics."""
        if self._is_dummy_request(request_ids):
            return

        if token_count is None:
            raise ValueError(
                "token_count is required for ModelPerformanceTracker")

        self.decode_metrics.add_measurement(latency, token_count)

    def print_final_stats(self):
        """Print final statistics with throughput information."""
        logger.info("=" * 80)
        logger.info("FINAL PERFORMANCE STATISTICS (MODEL)")
        logger.info("=" * 80)

        # Prefill stats
        if self.prefill_metrics.get_call_counts() > 0:
            logger.info("PREFILL METRICS:")
            logger.info("  Total call counts: %d",
                        self.prefill_metrics.get_call_counts())
            logger.info("  Total tokens processed: %d",
                        self.prefill_metrics.get_total_tokens())
            logger.info("  Average latency: %.2f ms",
                        self.prefill_metrics.get_avg_latency())
            logger.info("  Average throughput: %.2f tokens/sec",
                        self.prefill_metrics.get_avg_throughput())
            if self.prefill_metrics_by_request_id.get_num_request_ids() > 0:
                avg_latency_by_request_id = \
                    self.prefill_metrics_by_request_id.get_avg_latency_per_request()
                logger.info("  Average latency per request:")
                for request_id, latency in avg_latency_by_request_id.items():
                    logger.info("    %s: %.2f ms", request_id, latency)
        else:
            logger.info("PREFILL METRICS: No data recorded")

        logger.info("-" * 40)

        # Decode stats
        if self.decode_metrics.get_call_counts() > 0:
            logger.info("DECODE METRICS:")
            logger.info("  Total call counts: %d",
                        self.decode_metrics.get_call_counts())
            logger.info("  Total tokens processed: %d",
                        self.decode_metrics.get_total_tokens())
            logger.info("  Average latency: %.2f ms",
                        self.decode_metrics.get_avg_latency())
            logger.info("  Average throughput: %.2f tokens/sec",
                        self.decode_metrics.get_avg_throughput())
        else:
            logger.info("DECODE METRICS: No data recorded")

        logger.info("=" * 80)


class SamplerPerformanceTracker(BasePerformanceTracker):
    """Tracks performance metrics with latency-only tracking for samplers."""

    def __init__(self):
        super().__init__()
        self.prefill_metrics = LatencyOnlyMetrics()
        self.decode_metrics = LatencyOnlyMetrics()
        self.prefill_metrics_by_request_id = PrefillMetricsByRequestID(
            LatencyOnlyMetrics)

    def record_prefill(self,
                       latency: float,
                       token_count: Optional[int] = None,
                       request_ids: Optional[list[str]] = None):
        """Record prefill step metrics (token_count is ignored)."""
        if self._is_dummy_request(request_ids):
            return

        request_id = None
        if request_ids is not None:
            assert len(request_ids) == 1, (
                f"Expected exactly one request_id during prefill, "
                f"got {len(request_ids)}: {request_ids}")
            request_id = request_ids[0]

        self.prefill_metrics.add_measurement(latency)
        if request_id:
            self.prefill_metrics_by_request_id.add_measurement(
                request_id, latency)

    def record_decode(self,
                      latency: float,
                      token_count: Optional[int] = None,
                      request_ids: Optional[list[str]] = None):
        """Record decode step metrics."""
        if self._is_dummy_request(request_ids):
            return

        self.decode_metrics.add_measurement(latency)

    def print_final_stats(self):
        """Print final statistics with latency-only information."""
        logger.info("=" * 80)
        logger.info("FINAL PERFORMANCE STATISTICS (SAMPLER)")
        logger.info("=" * 80)

        # Prefill stats
        if self.prefill_metrics.get_call_counts() > 0:
            logger.info("PREFILL METRICS:")
            logger.info("  Total call counts: %d",
                        self.prefill_metrics.get_call_counts())
            logger.info("  Average latency: %.2f ms",
                        self.prefill_metrics.get_avg_latency())
            if self.prefill_metrics_by_request_id.get_num_request_ids() > 0:
                avg_latency_by_request_id = \
                    self.prefill_metrics_by_request_id.get_avg_latency_per_request()
                logger.info("  Average latency per request:")
                for request_id, latency in avg_latency_by_request_id.items():
                    logger.info("    %s: %.2f ms", request_id, latency)
        else:
            logger.info("PREFILL METRICS: No data recorded")

        logger.info("-" * 40)

        # Decode stats
        if self.decode_metrics.get_call_counts() > 0:
            logger.info("DECODE METRICS:")
            logger.info("  Total call counts: %d",
                        self.decode_metrics.get_call_counts())
            logger.info("  Average latency: %.2f ms",
                        self.decode_metrics.get_avg_latency())
        else:
            logger.info("DECODE METRICS: No data recorded")

        logger.info("=" * 80)
