import time
import atexit
from dataclasses import dataclass, field
from collections import defaultdict

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
    
    def get_avg_latency(self) -> float:
        """Get average latency in milliseconds."""
        return sum(self.latencies) / len(self.latencies) * 1000 if self.latencies else 0.0
    
    def get_avg_throughput(self) -> float:
        """Get average throughput in tokens/second."""
        if not self.latencies or not self.token_counts:
            return 0.0
        total_time = sum(self.latencies)
        total_tokens = sum(self.token_counts)
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
            logger.info("  Total call counts: %d", self.prefill_metrics.get_call_counts())
            logger.info("  Total tokens processed: %d", sum(self.prefill_metrics.token_counts))
            logger.info("  Average latency: %.2f ms", self.prefill_metrics.get_avg_latency())
            logger.info("  Average throughput: %.2f tokens/sec", self.prefill_metrics.get_avg_throughput())
        else:
            logger.info("PREFILL METRICS: No data recorded")
        
        logger.info("-" * 40)
        
        # Decode stats
        if self.decode_metrics.get_call_counts() > 0:
            logger.info("DECODE METRICS:")
            logger.info("  Total call counts: %d", self.decode_metrics.get_call_counts())
            logger.info("  Total tokens processed: %d", sum(self.decode_metrics.token_counts))
            logger.info("  Average latency: %.2f ms", self.decode_metrics.get_avg_latency())
            logger.info("  Average throughput: %.2f tokens/sec", self.decode_metrics.get_avg_throughput())
        else:
            logger.info("DECODE METRICS: No data recorded")
        
        logger.info("=" * 80)
