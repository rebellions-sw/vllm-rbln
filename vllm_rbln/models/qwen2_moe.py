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
import torch.nn.functional as F
from vllm.distributed import tensor_model_parallel_all_reduce
from vllm.model_executor.models.qwen2_moe import Qwen2MoeSparseMoeBlock


def __qwen2_moe_forward_rsd(self, hidden_states: torch.Tensor) -> torch.Tensor:
    shared_output = None
    if self.shared_expert is not None:
        shared_output = self.shared_expert(hidden_states)
        if self.shared_expert_gate is not None:
            shared_output = F.sigmoid(
                self.shared_expert_gate(hidden_states)) * shared_output
    router_logits, _ = self.gate(hidden_states)
    final_hidden_states = self.experts(hidden_states=hidden_states,
                                       router_logits=router_logits)
    if shared_output is not None:
        final_hidden_states = final_hidden_states + shared_output
    if self.tp_size > 1:
        final_hidden_states = tensor_model_parallel_all_reduce(
            final_hidden_states)
    return final_hidden_states


Qwen2MoeSparseMoeBlock.forward = __qwen2_moe_forward_rsd
