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
from dataclasses import dataclass, field
from typing import List, Optional, TypeVar

from vllm_rbln.logger import init_logger

logger = init_logger(__name__)

T = TypeVar('T', int, float)


def _remove_outlier(values: List[T]) -> List[T]:
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
    latencies: List[float] = field(default_factory=list)
    host_times: List[int] = field(default_factory=list)
    device_times: List[int] = field(default_factory=list)
    ccl_times: List[int] = field(default_factory=list)

    def get_call_counts(self) -> int:
        """Get total number of calls processed."""
        return len(self.latencies)

    def get_avg_latency(self, ignore_outlier: bool = True) -> float:
        """Get average latency in milliseconds,
        optionally ignoring one outlier."""
        values = _remove_outlier(
            self.latencies) if ignore_outlier else self.latencies
        return sum(values) / len(values) * 1000 if values else 0.0

    def get_avg_host_time(self, ignore_outlier: bool = True) -> float:
        """Get average host time in microseconds,
        optionally ignoring one outlier."""
        values = _remove_outlier(
            self.host_times) if ignore_outlier else self.host_times
        return sum(values) / len(values) if values else 0.0

    def get_avg_device_time(self, ignore_outlier: bool = True) -> float:
        """Get average device time in microseconds,
        optionally ignoring one outlier."""
        values = _remove_outlier(
            self.device_times) if ignore_outlier else self.device_times
        return sum(values) / len(values) if values else 0.0

    def get_avg_ccl_time(self, ignore_outlier: bool = True) -> float:
        """Get average ccl time in microseconds,
        optionally ignoring one outlier."""
        values = _remove_outlier(
            self.ccl_times) if ignore_outlier else self.ccl_times
        return sum(values) / len(values) if values else 0.0


@dataclass
class LatencyOnlyMetrics(BaseStepMetrics):
    """Metrics that only track latency (for samplers)."""

    def add_measurement(
        self,
        latency: float,
        host_time: Optional[int] = None,
        device_time: Optional[int] = None,
        ccl_time: Optional[int] = None,
    ):
        """Add a latency measurement."""
        self.latencies.append(latency)
        if host_time is not None:
            self.host_times.append(host_time)
        if device_time is not None:
            self.device_times.append(device_time)
        if ccl_time is not None:
            self.ccl_times.append(ccl_time)


@dataclass
class ThroughputMetrics(BaseStepMetrics):
    """Metrics that track both latency and throughput (for models)."""
    token_counts: List[int] = field(default_factory=list)

    def add_measurement(
        self,
        latency: float,
        token_count: int,
        host_time: Optional[int] = None,
        device_time: Optional[int] = None,
        ccl_time: Optional[int] = None,
    ):
        """Add a latency and token count measurement."""
        self.latencies.append(latency)
        self.token_counts.append(token_count)
        if host_time is not None:
            self.host_times.append(host_time)
        if device_time is not None:
            self.device_times.append(device_time)
        if ccl_time is not None:
            self.ccl_times.append(ccl_time)

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
    def _is_dummy_request(request_ids: Optional[List[str]]) -> bool:
        """Check if the request is a dummy request."""
        if request_ids and request_ids[0].startswith("dummy_request_"):
            return True
        return False

    @abstractmethod
    def record_prefill(
        self,
        latency: float,
        token_count: Optional[int] = None,
        request_ids: Optional[List[str]] = None,
        host_time: Optional[int] = None,
        device_time: Optional[int] = None,
        ccl_time: Optional[int] = None,
    ):
        """Record prefill step metrics."""
        pass

    @abstractmethod
    def record_decode(
        self,
        latency: float,
        token_count: Optional[int] = None,
        request_ids: Optional[List[str]] = None,
        host_time: Optional[int] = None,
        device_time: Optional[int] = None,
        ccl_time: Optional[int] = None,
        padded_decode: bool = False,
    ):
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
        self.padded_decode_metrics = ThroughputMetrics()

    def record_prefill(
        self,
        latency: float,
        token_count: Optional[int] = None,
        request_ids: Optional[List[str]] = None,
        host_time: Optional[int] = None,
        device_time: Optional[int] = None,
        ccl_time: Optional[int] = None,
    ):
        """Record prefill step metrics."""
        if self._is_dummy_request(request_ids):
            return

        if token_count is None:
            raise ValueError(
                "token_count is required for ModelPerformanceTracker")

        self.prefill_metrics.add_measurement(latency, token_count, host_time,
                                             device_time, ccl_time)

    def record_decode(
        self,
        latency: float,
        token_count: Optional[int] = None,
        request_ids: Optional[List[str]] = None,
        host_time: Optional[int] = None,
        device_time: Optional[int] = None,
        ccl_time: Optional[int] = None,
        padded_decode: bool = False,
    ):
        """Record decode step metrics."""
        if self._is_dummy_request(request_ids):
            return

        if token_count is None:
            raise ValueError(
                "token_count is required for ModelPerformanceTracker")

        metrics = self.padded_decode_metrics if padded_decode \
            else self.decode_metrics
        metrics.add_measurement(latency, token_count, host_time, device_time,
                                ccl_time)

    def print_final_stats(self):
        """Print final statistics with throughput information."""
        logger.info("=" * 80)
        logger.info(f"FINAL PERFORMANCE STATISTICS (MODEL)")
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
            if self.prefill_metrics.host_times:
                logger.info("  Average host time: %.2f us",
                            self.prefill_metrics.get_avg_host_time())
            if self.prefill_metrics.device_times:
                logger.info("  Average device time: %.2f us",
                            self.prefill_metrics.get_avg_device_time())
            if self.prefill_metrics.ccl_times:
                logger.info("  Average ccl time: %.2f us",
                            self.prefill_metrics.get_avg_ccl_time())
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
            if self.decode_metrics.host_times:
                logger.info("  Average host time: %.2f us",
                            self.decode_metrics.get_avg_host_time())
            if self.decode_metrics.device_times:
                logger.info("  Average device time: %.2f us",
                            self.decode_metrics.get_avg_device_time())
            if self.decode_metrics.ccl_times:
                logger.info("  Average ccl time: %.2f us",
                            self.decode_metrics.get_avg_ccl_time())
        else:
            logger.info("DECODE METRICS: No data recorded")

        logger.info("-" * 40)

        # Padded decode stats
        if self.padded_decode_metrics.get_call_counts() > 0:
            logger.info("PADDED DECODE METRICS:")
            logger.info("  Total call counts: %d",
                        self.padded_decode_metrics.get_call_counts())
            logger.info("  Total tokens processed: %d",
                        self.padded_decode_metrics.get_total_tokens())
            logger.info("  Average latency: %.2f ms",
                        self.padded_decode_metrics.get_avg_latency())
            logger.info("  Average throughput: %.2f tokens/sec",
                        self.padded_decode_metrics.get_avg_throughput())
            if self.padded_decode_metrics.host_times:
                logger.info("  Average host time: %.2f us",
                            self.padded_decode_metrics.get_avg_host_time())
            if self.padded_decode_metrics.device_times:
                logger.info("  Average device time: %.2f us",
                            self.padded_decode_metrics.get_avg_device_time())
            if self.padded_decode_metrics.ccl_times:
                logger.info("  Average ccl time: %.2f us",
                            self.padded_decode_metrics.get_avg_ccl_time())
        else:
            logger.info("PADDED DECODE METRICS: No data recorded")

        logger.info("=" * 80)


class SamplerPerformanceTracker(BasePerformanceTracker):
    """Tracks performance metrics with latency-only tracking for samplers."""

    def __init__(self):
        super().__init__()
        self.prefill_metrics = LatencyOnlyMetrics()
        self.decode_metrics = LatencyOnlyMetrics()
        self.padded_decode_metrics = LatencyOnlyMetrics()

    def record_prefill(
        self,
        latency: float,
        token_count: Optional[int] = None,
        request_ids: Optional[List[str]] = None,
        host_time: Optional[int] = None,
        device_time: Optional[int] = None,
        ccl_time: Optional[int] = None,
    ):
        """Record prefill step metrics (token_count is ignored)."""
        if self._is_dummy_request(request_ids):
            return

        self.prefill_metrics.add_measurement(latency, host_time, device_time,
                                             ccl_time)

    def record_decode(
        self,
        latency: float,
        token_count: Optional[int] = None,
        request_ids: Optional[List[str]] = None,
        host_time: Optional[int] = None,
        device_time: Optional[int] = None,
        ccl_time: Optional[int] = None,
        padded_decode: bool = False,
    ):
        """Record decode step metrics (token_count is ignored)."""
        if self._is_dummy_request(request_ids):
            return

        metrics = self.padded_decode_metrics if padded_decode \
            else self.decode_metrics
        metrics.add_measurement(latency, host_time, device_time, ccl_time)

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
            if self.prefill_metrics.host_times:
                logger.info("  Average host time: %.2f us",
                            self.prefill_metrics.get_avg_host_time())
            if self.prefill_metrics.device_times:
                logger.info("  Average device time: %.2f us",
                            self.prefill_metrics.get_avg_device_time())
            if self.prefill_metrics.ccl_times:
                logger.info("  Average ccl time: %.2f us",
                            self.prefill_metrics.get_avg_ccl_time())
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
            if self.decode_metrics.host_times:
                logger.info("  Average host time: %.2f us",
                            self.decode_metrics.get_avg_host_time())
            if self.decode_metrics.device_times:
                logger.info("  Average device time: %.2f us",
                            self.decode_metrics.get_avg_device_time())
            if self.decode_metrics.ccl_times:
                logger.info("  Average ccl time: %.2f us",
                            self.decode_metrics.get_avg_ccl_time())
        else:
            logger.info("DECODE METRICS: No data recorded")

        logger.info("-" * 40)

        # Padded decode stats
        if self.padded_decode_metrics.get_call_counts() > 0:
            logger.info("PADDED DECODE METRICS:")
            logger.info("  Total call counts: %d",
                        self.padded_decode_metrics.get_call_counts())
            logger.info("  Average latency: %.2f ms",
                        self.padded_decode_metrics.get_avg_latency())
            if self.padded_decode_metrics.host_times:
                logger.info("  Average host time: %.2f us",
                            self.padded_decode_metrics.get_avg_host_time())
            if self.padded_decode_metrics.device_times:
                logger.info("  Average device time: %.2f us",
                            self.padded_decode_metrics.get_avg_device_time())
            if self.padded_decode_metrics.ccl_times:
                logger.info("  Average ccl time: %.2f us",
                            self.padded_decode_metrics.get_avg_ccl_time())
        else:
            logger.info("PADDED DECODE METRICS: No data recorded")

        logger.info("=" * 80)
