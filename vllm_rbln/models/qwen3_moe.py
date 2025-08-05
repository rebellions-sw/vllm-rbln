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
from typing import Optional, Union
from vllm.distributed import (tensor_model_parallel_all_reduce,
                              get_pp_group)
from vllm.model_executor.models.qwen3_moe import (Qwen3MoeSparseMoeBlock,
                                                  Qwen3MoeDecoderLayer,
                                                  Qwen3MoeModel,
                                                  Qwen3MoeForCausalLM)
from vllm.sequence import IntermediateTensors


def __qwen3_moe_sparse_moe_block_forward_rsd(self, hidden_states: torch.Tensor) -> torch.Tensor:
    # NOTE: hidden_states can have either 1D or 2D shape.
    # router_logits: (num_tokens, n_experts)
    router_logits, _ = self.gate(hidden_states)
    final_hidden_states, selected_experts = self.experts(hidden_states=hidden_states,
                                       router_logits=router_logits)
    final_hidden_states = final_hidden_states
    if self.tp_size > 1:
        final_hidden_states = tensor_model_parallel_all_reduce(
            final_hidden_states)

    return final_hidden_states, selected_experts


def __qwen3_moe_decoder_layer_forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: Optional[torch.Tensor],
    ) -> torch.Tensor:
        # Self Attention
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(
                hidden_states, residual)
        hidden_states = self.self_attn(
            positions=positions,
            hidden_states=hidden_states,
        )

        # Fully Connected
        hidden_states, residual = self.post_attention_layernorm(
            hidden_states, residual)
        hidden_states, selected_experts = self.mlp(hidden_states)
        return hidden_states, residual, selected_experts


def __qwen3_moe_model_forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: Optional[IntermediateTensors] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, IntermediateTensors]:
        if get_pp_group().is_first_rank:
            if inputs_embeds is not None:
                hidden_states = inputs_embeds
            else:
                hidden_states = self.get_input_embeddings(input_ids)
            residual = None
        else:
            assert intermediate_tensors is not None
            hidden_states = intermediate_tensors["hidden_states"]
            residual = intermediate_tensors["residual"]
        selected_experts_list = []
        for i in range(self.start_layer, self.end_layer):
            layer = self.layers[i]
            hidden_states, residual, selected_experts = layer(positions, hidden_states, residual)
            selected_experts_list.append(selected_experts)
        if not get_pp_group().is_last_rank:
            return IntermediateTensors({
                "hidden_states": hidden_states,
                "residual": residual
            })
        hidden_states, _ = self.norm(hidden_states, residual)
        return hidden_states, selected_experts_list

def __qwen3_moe_causal_forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: Optional[IntermediateTensors] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, IntermediateTensors]:
        hidden_states, selected_experts = self.model(input_ids, positions, intermediate_tensors,
                                   inputs_embeds)
        return hidden_states, selected_experts


Qwen3MoeSparseMoeBlock.forward = __qwen3_moe_sparse_moe_block_forward_rsd
Qwen3MoeDecoderLayer.forward = __qwen3_moe_decoder_layer_forward
Qwen3MoeModel.forward = __qwen3_moe_model_forward
Qwen3MoeForCausalLM.forward = __qwen3_moe_causal_forward
