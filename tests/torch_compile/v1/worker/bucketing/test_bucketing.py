# Copyright 2025 Rebellions Inc. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

import pytest

from vllm_rbln.v1.worker.bucketing import (
    ExponentialBucketingManager,
    LinearBucketingManager,
    get_bucketing_manager_class,
)


# --- Factory function tests ---


def test_get_bucketing_manager_class_exponential():
    assert (
        get_bucketing_manager_class("exponential")
        is ExponentialBucketingManager
    )
    assert get_bucketing_manager_class("exp") is ExponentialBucketingManager


def test_get_bucketing_manager_class_linear():
    assert get_bucketing_manager_class("linear") is LinearBucketingManager


def test_get_bucketing_manager_class_invalid():
    with pytest.raises(ValueError, match="Invalid bucketing strategy"):
        get_bucketing_manager_class("unknown")


# --- Base class validation tests ---


def test_max_lt_min_raises():
    with pytest.raises(ValueError, match="max_batch_size must be >= min"):
        LinearBucketingManager(max_batch_size=2, min_batch_size=10)


def test_limit_zero_raises():
    with pytest.raises(ValueError, match="limit must be greater than 0"):
        LinearBucketingManager(max_batch_size=32, limit=0)


def test_step_zero_raises():
    with pytest.raises(ValueError, match="step must be greater than 0"):
        LinearBucketingManager(max_batch_size=32, step=0)


def test_min_batch_size_zero_raises():
    with pytest.raises(ValueError, match="min_batch_size must be greater"):
        LinearBucketingManager(max_batch_size=32, min_batch_size=0)


# --- LinearBucketingManager tests ---


def test_linear_basic():
    mgr = LinearBucketingManager(max_batch_size=10, min_batch_size=1, step=2)
    # Should produce descending: 10, 8, 6, 4, 2 → sorted
    assert mgr.decode_batch_buckets == sorted(mgr.decode_batch_buckets)
    assert mgr.decode_batch_buckets[0] >= 1
    assert mgr.decode_batch_buckets[-1] == 10


def test_linear_single_bucket():
    mgr = LinearBucketingManager(max_batch_size=5, min_batch_size=5, step=2)
    assert mgr.decode_batch_buckets == [5]


def test_linear_batch_buckets_include_prefill():
    mgr = LinearBucketingManager(max_batch_size=8, min_batch_size=2, step=2)
    assert 1 in mgr.batch_buckets, "Prefill bucket (1) must be included"


def test_linear_respects_limit():
    mgr = LinearBucketingManager(
        max_batch_size=1000, min_batch_size=1, step=1, limit=5
    )
    assert len(mgr.decode_batch_buckets) <= 5


# --- ExponentialBucketingManager tests ---


def test_exponential_basic():
    mgr = ExponentialBucketingManager(
        max_batch_size=32, min_batch_size=1, step=2
    )
    # 32, 16, 8, 4, 2, 1
    assert mgr.decode_batch_buckets == [1, 2, 4, 8, 16, 32]


def test_exponential_step_must_be_gt_1():
    with pytest.raises(ValueError, match="step must be greater than 1"):
        ExponentialBucketingManager(max_batch_size=32, step=1)


def test_exponential_single_bucket():
    mgr = ExponentialBucketingManager(
        max_batch_size=4, min_batch_size=4, step=2
    )
    assert mgr.decode_batch_buckets == [4]


def test_exponential_batch_buckets_include_prefill():
    mgr = ExponentialBucketingManager(
        max_batch_size=16, min_batch_size=2, step=2
    )
    assert 1 in mgr.batch_buckets


def test_exponential_respects_limit():
    mgr = ExponentialBucketingManager(
        max_batch_size=2**20, min_batch_size=1, step=2, limit=3
    )
    assert len(mgr.decode_batch_buckets) <= 3


# --- find_decode_batch_bucket tests ---


def test_find_decode_batch_bucket():
    mgr = ExponentialBucketingManager(
        max_batch_size=32, min_batch_size=1, step=2
    )
    assert mgr.find_decode_batch_bucket(1) == 1
    assert mgr.find_decode_batch_bucket(3) == 4
    assert mgr.find_decode_batch_bucket(32) == 32


def test_find_decode_batch_bucket_no_fit():
    mgr = ExponentialBucketingManager(
        max_batch_size=8, min_batch_size=1, step=2
    )
    with pytest.raises(ValueError, match="No batch bucket found"):
        mgr.find_decode_batch_bucket(100)
