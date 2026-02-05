# Copyright 2025 Rebellions Inc. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

import os

from vllm_rbln.v1.worker.utils import set_omp_num_threads


def test_set_omp_num_threads_default(monkeypatch):
    """When OMP_NUM_THREADS is not set, it should be set to default."""
    monkeypatch.delenv("OMP_NUM_THREADS", raising=False)
    set_omp_num_threads(rank=0, local_rank=0, default_num_threads=4)
    assert os.environ["OMP_NUM_THREADS"] == "4"


def test_set_omp_num_threads_existing(monkeypatch):
    """When OMP_NUM_THREADS is already set, it should not be changed."""
    monkeypatch.setenv("OMP_NUM_THREADS", "8")
    set_omp_num_threads(rank=0, local_rank=0, default_num_threads=2)
    assert os.environ["OMP_NUM_THREADS"] == "8"


def test_set_cpu_affinity_nobind_when_numa_disabled(monkeypatch):
    """When VLLM_RBLN_NUMA is False, affinity should be nobind (no-op)."""
    from unittest.mock import MagicMock

    from vllm_rbln.v1.worker import utils as worker_utils

    monkeypatch.setattr(worker_utils.envs, "VLLM_RBLN_NUMA", False)
    parallel_cfg = MagicMock()

    # Should not raise; effectively a no-op
    worker_utils.set_cpu_affinity(
        rank=0, local_rank=0, parallel_config=parallel_cfg
    )
