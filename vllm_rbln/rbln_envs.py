# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os
from vllm.envs import environment_variables as vllm_envs
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    RBLN_COMPILE_MODEL: bool = True
    RBLN_TP_SIZE: int = 1


# extended environments
environment_variables = {
    **vllm_envs,
    # If true, will compile models using torch.compile.
    # Otherwise, run the CPU eager mode, if possible.
    "RBLN_COMPILE_MODEL": (
        lambda: os.environ.get("COMPILE_MODEL", "True").lower() in ("true", "1")
    ),
    # TP Size for RSD.
    "RBLN_TP_SIZE": lambda: int(os.environ.get("TP_SIZE", 1)),
}


def __getattr__(name: str):
    # lazy evaluation of environment variables
    if name in environment_variables:
        return environment_variables[name]()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
