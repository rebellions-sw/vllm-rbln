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
from vllm.distributed import (
    get_tensor_model_parallel_world_size,
    tensor_model_parallel_all_reduce,
)
from vllm.model_executor.layers.fused_moe.layer import FusedMoE
from vllm.model_executor.layers.fused_moe.shared_fused_moe import (
    SharedFusedMoE)

from vllm_rbln.logger import init_logger

logger = init_logger(__name__)


def __shared_fused_moe_init_rbln(
    self: SharedFusedMoE,
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


def __shared_fused_moe_forward_rbln(
    self: SharedFusedMoE, hidden_states: torch.Tensor, router: torch.nn.Module
) -> tuple[torch.Tensor, torch.Tensor]:
    if not self.use_overlapped:
        if self._shared_experts is not None:
            shared_out = self._shared_experts(hidden_states)

            # Reduce shared expert outputs if necessary, since the MLP
            # should have been created with reduce_results=False.
            if (
                self.reduce_results
                and get_tensor_model_parallel_world_size() > 1
                and self.must_reduce_shared_expert_outputs()
            ):
                shared_out = tensor_model_parallel_all_reduce(shared_out)
        else:
            shared_out = None

        fused_out = FusedMoE.forward_oot(
            self,
            hidden_states=hidden_states,
            router=router,
        )
    else:
        shared_out, fused_out = FusedMoE.forward(
            self,
            hidden_states=hidden_states,
            router=router,
        )
        # ensure early TP reduction of shared expert outputs when required
        if (
            shared_out is not None
            and self.reduce_results
            and get_tensor_model_parallel_world_size() > 1
            and self.must_reduce_shared_expert_outputs()
        ):
            shared_out = tensor_model_parallel_all_reduce(shared_out)
    return shared_out, fused_out


SharedFusedMoE.__init__ = __shared_fused_moe_init_rbln
SharedFusedMoE.forward_oot = __shared_fused_moe_forward_rbln
# Remove upstream SharedFusedMoE.forward override so that CustomOp.forward
# dispatches to forward_oot.
del SharedFusedMoE.forward
