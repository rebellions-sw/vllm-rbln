# Copyright 2025 Rebellions Inc. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

from vllm_rbln.worker.metrics import (
    PrefillMetricsByRequestID,
    PerformanceTracker,
    StepMetrics,
)


# --- StepMetrics tests ---


def test_step_metrics_add_measurement():
    m = StepMetrics()
    m.add_measurement(0.1, 10, host_time=100, device_time=200, ccl_time=50)
    assert len(m.latencies) == 1
    assert len(m.token_counts) == 1
    assert len(m.host_times) == 1
    assert len(m.device_times) == 1
    assert len(m.ccl_times) == 1


def test_step_metrics_add_measurement_optional():
    m = StepMetrics()
    m.add_measurement(0.5, 20)
    assert len(m.latencies) == 1
    assert len(m.host_times) == 0
    assert len(m.device_times) == 0
    assert len(m.ccl_times) == 0


def test_step_metrics_get_call_counts():
    m = StepMetrics()
    assert m.get_call_counts() == 0
    m.add_measurement(0.1, 10)
    m.add_measurement(0.2, 20)
    assert m.get_call_counts() == 2


def test_step_metrics_avg_latency():
    m = StepMetrics()
    m.add_measurement(0.1, 10)
    m.add_measurement(0.2, 20)
    m.add_measurement(0.15, 15)
    avg = m.get_avg_latency(ignore_outlier=False)
    expected = (0.1 + 0.2 + 0.15) / 3 * 1000
    assert abs(avg - expected) < 0.01


def test_step_metrics_avg_latency_with_outlier():
    m = StepMetrics()
    m.add_measurement(0.1, 10)
    m.add_measurement(0.1, 10)
    m.add_measurement(10.0, 10)  # outlier
    avg_with = m.get_avg_latency(ignore_outlier=True)
    avg_without = m.get_avg_latency(ignore_outlier=False)
    assert avg_with < avg_without


def test_step_metrics_avg_latency_empty():
    m = StepMetrics()
    assert m.get_avg_latency() == 0.0


def test_step_metrics_avg_throughput():
    m = StepMetrics()
    m.add_measurement(1.0, 100)
    m.add_measurement(1.0, 100)
    tp = m.get_avg_throughput(ignore_outlier=False)
    assert tp == 100.0  # 200 tokens / 2.0 seconds


def test_step_metrics_avg_throughput_empty():
    m = StepMetrics()
    assert m.get_avg_throughput() == 0.0


def test_step_metrics_avg_host_device_ccl_time():
    m = StepMetrics()
    m.add_measurement(0.1, 10, host_time=100, device_time=200, ccl_time=50)
    m.add_measurement(0.1, 10, host_time=300, device_time=400, ccl_time=150)
    assert m.get_avg_host_time(ignore_outlier=False) == 200.0
    assert m.get_avg_device_time(ignore_outlier=False) == 300.0
    assert m.get_avg_ccl_time(ignore_outlier=False) == 100.0


def test_step_metrics_avg_times_empty():
    m = StepMetrics()
    assert m.get_avg_host_time() == 0.0
    assert m.get_avg_device_time() == 0.0
    assert m.get_avg_ccl_time() == 0.0


def test_without_outlier_single_value():
    m = StepMetrics()
    assert m._without_outlier_f([5.0]) == [5.0]
    assert m._without_outlier_i([5]) == [5]


def test_without_outlier_empty():
    m = StepMetrics()
    assert m._without_outlier_f([]) == []
    assert m._without_outlier_i([]) == []


# --- PrefillMetricsByRequestID tests ---


def test_prefill_by_request_id():
    pm = PrefillMetricsByRequestID()
    pm.add_measurement("req-1", 0.1, 10)
    pm.add_measurement("req-1", 0.2, 20)
    pm.add_measurement("req-2", 0.3, 30)

    assert pm.get_num_request_ids() == 2
    avg = pm.get_avg_latency_per_request()
    assert "req-1" in avg
    assert "req-2" in avg


# --- PerformanceTracker tests ---


def test_performance_tracker_record_prefill():
    tracker = PerformanceTracker()
    tracker.record_prefill(0.1, 10, request_ids=["req-1"])
    assert tracker.prefill_metrics.get_call_counts() == 1


def test_performance_tracker_record_decode():
    tracker = PerformanceTracker()
    tracker.record_decode(0.1, 10)
    assert tracker.decode_metrics.get_call_counts() == 1


def test_performance_tracker_record_padded_decode():
    tracker = PerformanceTracker()
    tracker.record_decode(0.1, 10, padded_decode=True)
    assert tracker.padded_decode_metrics.get_call_counts() == 1
    assert tracker.decode_metrics.get_call_counts() == 0


def test_performance_tracker_check_dummy_request():
    tracker = PerformanceTracker()
    assert tracker.check_dummy_request(["dummy_request_0"]) is True
    assert tracker.check_dummy_request(["real_request_1"]) is False
    assert tracker.check_dummy_request(None) is False
    assert tracker.check_dummy_request([]) is False


def test_performance_tracker_skip_dummy():
    tracker = PerformanceTracker()
    tracker.record_prefill(0.1, 10, request_ids=["dummy_request_0"])
    assert tracker.prefill_metrics.get_call_counts() == 0

    tracker.record_decode(0.1, 10, request_ids=["dummy_request_1"])
    assert tracker.decode_metrics.get_call_counts() == 0


def test_performance_tracker_register_cleanup():
    tracker = PerformanceTracker()
    assert not tracker._registered_cleanup
    tracker.register_cleanup()
    assert tracker._registered_cleanup
    # Second call should be idempotent
    tracker.register_cleanup()
    assert tracker._registered_cleanup
