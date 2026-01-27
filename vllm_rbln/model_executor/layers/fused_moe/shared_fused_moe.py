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

import torch
from vllm.model_executor.layers.fused_moe.layer import FusedMoE
from vllm.model_executor.layers.fused_moe.shared_fused_moe import SharedFusedMoE

import vllm_rbln.rbln_envs as envs
from vllm_rbln.logger import init_logger

logger = init_logger(__name__)


def __shared_fused_moe_init_rbln(
    self,
    shared_experts: torch.nn.Module | None,
    gate: torch.nn.Module | None = None,
    use_overlapped: bool = True,
    **kwargs,
):
    FusedMoE.__init__(self, **kwargs)
    self._shared_experts = shared_experts

    # FIXME(RBLN) - disable use overlapped, not supported
    self.use_overlapped = False

    self._gate = gate


SharedFusedMoE.__init__ = __shared_fused_moe_init_rbln
