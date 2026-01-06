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

import tempfile

import pytest
import torch


@pytest.fixture(autouse=True)
def fresh_inductor_cache_per_test(monkeypatch):
    cache_dir = tempfile.mkdtemp(prefix="torchinductor_pytest_", dir="/tmp")
    monkeypatch.setenv("TORCHINDUCTOR_CACHE_DIR", cache_dir)

    # Dynamo 인프로세스 캐시도 같이 리셋 (재사용 최소화)
    torch._dynamo.reset()

    yield
