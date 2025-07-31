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

import os
from typing import TYPE_CHECKING

from vllm.envs import environment_variables as vllm_envs

if TYPE_CHECKING:
    RBLN_COMPILE_MODEL: bool = True
    RBLN_TP_SIZE: int = 1
    RBLN_SAMPLER: bool = False
    RBLN_ENABLE_WARM_UP: bool = False

# extended environments
environment_variables = {
    **vllm_envs,
    # If true, will compile models using torch.compile.
    # Otherwise, run the CPU eager mode, if possible.
    "RBLN_COMPILE_MODEL":
    (lambda: os.environ.get("COMPILE_MODEL", "True").lower() in ("true", "1")),
    # TP Size for RSD.
    "RBLN_TP_SIZE":
    lambda: int(os.environ.get("TP_SIZE", 1)),
    # Use customized sampler
    "RBLN_SAMPLER":
    (lambda: os.environ.get("VLLM_RBLN_SAMPLER", "False").lower() in
     ("true", "1")),
    # Enable warmup
    "RBLN_ENABLE_WARM_UP":
    (lambda: os.environ.get("VLLM_RBLN_ENABLE_WARM_UP", "False").lower() in
     ("true", "1")),
}


def __getattr__(name: str):
    # lazy evaluation of environment variables
    if name in environment_variables:
        return environment_variables[name]()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
