import os, tempfile, shutil
from pathlib import Path
import pytest
import torch

@pytest.fixture(autouse=True)
def fresh_inductor_cache_per_test(monkeypatch):
    cache_dir = tempfile.mkdtemp(prefix="torchinductor_pytest_", dir="/tmp")
    monkeypatch.setenv("TORCHINDUCTOR_CACHE_DIR", cache_dir)

    # Dynamo 인프로세스 캐시도 같이 리셋 (재사용 최소화)
    torch._dynamo.reset()

    yield