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

from typing import Callable, Optional

import torch
import torch.nn.functional as F
from vllm.distributed import get_dp_group
from vllm.forward_context import get_forward_context
from vllm.model_executor.layers.fused_moe.layer import (
    FusedMoE, UnquantizedFusedMoEMethod)

import vllm_rbln.rbln_envs as envs
from vllm_rbln.logger import init_logger

logger = init_logger(__name__)

@torch.library.custom_op(
    "rbln_custom_ops::custom_moe_glu",
    mutates_args=(),
)
def custom_moe_glu(
    hidden_states: torch.Tensor,
    gate_proj_weight: torch.Tensor,
    up_proj_weight: torch.Tensor,
    down_proj_weight: torch.Tensor,
    masked_routing_weight: torch.Tensor,
    # act_fn: str,
    expert_select_count: torch.Tensor,
    gate_proj_bias: Optional[torch.Tensor] = None,
    up_proj_bias: Optional[torch.Tensor] = None,
    down_proj_bias: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Customized MoE GLU operation.

    Expected tensor shapes:
    - hidden_states: [batch *seq_len, hidden_size]
    - gate_proj_weight: [hidden_size, num_experts * intermediate_size]
    - up_proj_weight: [hidden_size, num_experts * intermediate_size]
    - down_proj_weight: [num_experts * intermediate_size, hidden_size]
    - masked_routing_weight: [batch * seq_len, num_experts]

    Returns:
        torch.Tensor: [batch * seq_len, hidden_size]
    """

    out = torch.zeros_like(hidden_states)
    expert_cnt = gate_proj_weight.shape[0]
    for i in range(expert_cnt):
        gate = torch.nn.functional.linear(hidden_states, gate_proj_weight[i])
        up = torch.nn.functional.linear(hidden_states, up_proj_weight[i])
        mul = torch.nn.functional.silu(gate) * up
        down = torch.nn.functional.linear(mul, down_proj_weight[i])
        out += down * masked_routing_weight[:, i:i+1]

    return out


@custom_moe_glu.register_fake
def custom_moe_glu_fake(
    hidden_states: torch.Tensor,
    gate_proj_weight: torch.Tensor,
    up_proj_weight: torch.Tensor,
    down_proj_weight: torch.Tensor,
    masked_routing_weight: torch.Tensor,
    expert_select_count: torch.Tensor,
    # act_fn: ACT_TYPES,
    gate_proj_bias: Optional[torch.Tensor] = None,
    up_proj_bias: Optional[torch.Tensor] = None,
    down_proj_bias: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    return torch.empty_like(hidden_states)


def unquantized_fused_moe_method_forward_rbln_rsd(
    self,
    layer: torch.nn.Module,
    x: torch.Tensor,
    use_grouped_topk: bool,
    top_k: int,
    router_logits: torch.Tensor,
    renormalize: bool,
    topk_group: Optional[int] = None,
    num_expert_group: Optional[int] = None,
    global_num_experts: int = -1,
    expert_map: Optional[torch.Tensor] = None,
    custom_routing_function: Optional[Callable] = None,
    scoring_func: str = "softmax",
    e_score_correction_bias: Optional[torch.Tensor] = None,
    activation: str = "silu",
    apply_router_weight_on_input: bool = False,
    **kwargs,
):
    # selected_experts
    w1 = layer.w13_weight
    w2 = layer.w2_weight

    orig_shape = x.shape  # noqa: F841
    hidden_size = x.shape[-1]
    num_tokens = x.shape[:-1].numel()  # noqa: F841
    num_experts = w1.shape[0]
    intermediate_size = w2.shape[-1]
    dtype = x.dtype

    hidden_states = x
    gating_output = router_logits
    topk_weights = gating_output.softmax(dim=-1, dtype=torch.float)
    topk_weights = topk_weights.to(torch.float)
    topk_weights, selected_experts = topk_weights.topk(top_k, dim=-1)
    if renormalize:
        topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)

    if expert_map is not None:
        selected_experts = expert_map[selected_experts]

    final_hidden_states = None

    # 1. build expert_mask & expert_weights
    # 2. FFN
    # topk_weights, expert_weights, expert_mask.shape = [b, seq, top_k]
    # NOTE - convert for loop scalar operation into tensor compare

    # [1,num_tokens,hidden_size]
    hidden_states = hidden_states.reshape(1, num_tokens, -1)
    # [num_experts,1,1,1]
    expert_idx_array = torch.arange(0,
                                    num_experts).reshape(num_experts, 1, 1, 1)
    # [1,1,num_tokens,topk]
    selected_experts_array = selected_experts.reshape(-1, 1, num_tokens, top_k)
    # [num_experts,1,num_tokens,topk]
    expert_mask_array = selected_experts_array == expert_idx_array
    # [num_experts,1,num_tokens,topk]
    topk_weights_array = topk_weights.reshape(-1, 1, num_tokens, top_k)
    # [num_experts,1,num_tokens,1]
    expert_weights_array = (topk_weights_array * expert_mask_array).sum(
        dim=-1, keepdim=True)
    # [1,num_tokens,1]
    temp_expert_weights = expert_weights_array[0]
    # NOTE - make explicit dependence between hidden_states and expert_weights
    # [1,num_tokens,hidden_size]
    # [1,num_tokens,1] <- broadcast add
    hidden_states = hidden_states + temp_expert_weights - temp_expert_weights
    # [num_experts,1,num_tokens,1] -> [num_experts,1,num_tokens,hidden_size]
    hidden_states = hidden_states.to(dtype)
    expert_weights_array = expert_weights_array.broadcast_to(
        (num_experts, 1, num_tokens, hidden_size)).to(dtype)
    # solution1. make custom operation for expert loop
    # solution2. add dummy use of expert_weights_array
    for expert_idx in range(num_experts):
        expert_w1 = w1[expert_idx]
        expert_w2 = w2[expert_idx]
        expert_weights = expert_weights_array[expert_idx]
        x = F.linear(hidden_states, expert_w1)
        gate = F.silu(x[..., :intermediate_size])
        x = x[..., intermediate_size:] * gate
        x = F.linear(x, expert_w2)

        current_hidden_states = x * expert_weights
        if final_hidden_states is None:
            final_hidden_states = current_hidden_states
        else:
            final_hidden_states = final_hidden_states + current_hidden_states

    assert final_hidden_states is not None
    return final_hidden_states.reshape(orig_shape)


# based on custom fused moe expert kernel
def get_masked_routing_weights(router_logits, top_k, renormalize):
    # routing_weights: (batch * sequence_length, n_experts)
    routing_weights = torch.nn.functional.softmax(router_logits, dim=1, dtype=torch.float)

    # selected_experts: (batch * sequence_length, top_k)
    selected_weights, selected_experts = torch.topk(routing_weights, k=top_k, dim=-1)
    mask = torch.zeros_like(routing_weights, dtype=torch.float32)
    un_mask = torch.ones_like(selected_experts, dtype=torch.float32)
    mask.scatter_(1, selected_experts, un_mask)

    if renormalize:  # only diff with mixtral sparse moe block!
        routing_weights /= selected_weights.sum(dim=-1, keepdim=True)

    masked_routing_weights = routing_weights * mask

    ## get size per expert
    expert = router_logits.shape[1]
    zeros = torch.zeros(expert, dtype=torch.int32)
    ones = torch.ones_like(selected_experts.view(-1), dtype=torch.int32)
    expert_select_count = torch.scatter_add(zeros, dim=0, index=selected_experts.view(-1), src=ones)

    return masked_routing_weights, expert_select_count

def unquantized_fused_moe_method_forward_rbln_custom(
    self,
    layer: torch.nn.Module,
    x: torch.Tensor,
    use_grouped_topk: bool,
    top_k: int,
    router_logits: torch.Tensor,
    renormalize: bool,
    topk_group: Optional[int] = None,
    num_expert_group: Optional[int] = None,
    global_num_experts: int = -1,
    expert_map: Optional[torch.Tensor] = None,
    custom_routing_function: Optional[Callable] = None,
    scoring_func: str = "softmax",
    e_score_correction_bias: Optional[torch.Tensor] = None,
    activation: str = "silu",
    apply_router_weight_on_input: bool = False,
    **kwargs,
):
    # selected_experts
    # w1 : gate_proj, w2 : down_proj, w3 : up_proj
    orig_shape = x.shape  # noqa: F841
    num_tokens = orig_shape[:-1].numel()  # noqa: F841
    intermediate_size = layer.w2_weight.shape[-1]

    # w13_weight- gate_up_proj_weight, merged weight data for gate_proj(w1_weight) and up_proj (w3_weight)
    # w2_weight - down_proj
    # gate_proj_weight - first half, layer.w13_weight[:intermediate_size]
    # up_proj_weight - second half, layer.w13_weight[intermediate_size:]
    # down_proj_weights = layer.w2_weight
    gate_proj_weight = layer.w13_weight[:,:intermediate_size,:]
    up_proj_weight = layer.w13_weight[:,intermediate_size:,:]
    down_proj_weight = layer.w2_weight

    # expected tensor shape - [num_tokens, -1]
    hidden_states = x.reshape(num_tokens, -1)
    router_logits = router_logits.reshape(num_tokens, -1)

    # optimum-rbln/src/optimum/rbln/transformers/models/qwen3_moe/qwen3_moe_architecture.py
    masked_routing_weights, expert_select_count = get_masked_routing_weights(router_logits, top_k, renormalize)
    final_hidden_states = torch.ops.rbln_custom_ops.custom_moe_glu(
        hidden_states,
        gate_proj_weight,
        up_proj_weight,
        down_proj_weight,
        masked_routing_weights,
        expert_select_count)
    return final_hidden_states.reshape(orig_shape)


def fused_moe_forward_rbln(self, hidden_states: torch.Tensor,
                           router_logits: torch.Tensor):
    assert self.quant_method is not None

    if self.dp_size > 1:
        org_hidden_shape = hidden_states.shape

        # input broadcast, all DPs broadcast each hidden_states & router_logits into dp group
        # example) DP2, TP/EP2
        # dp_group = {{0, 2}, {1, 3}}
        # tp_group = {{0, 1}, {2, 3}}
        # 1. initially, each DP hidden_states = [1, 128, 1024]
        # 2. after multicast(multiple broadcast), all DPs hidden_states = [dp_size, 128, 1024]
        # - all DP ranks braodcast its input to process group
        # 3. DP x TP/EP expert parallel
        # ex) 0, 1, 2, 3 has its own hidden_states = [dp_size, 128, 1024]
        # 4. dp_group all reduce - {0+2}, {1+3}, {0+2}, {1+3}
        # 5. select each DP rank output
        # 6. to_group all reduce - {0+2+1+3}, {0+2+1+3}, {0+2+1+3}, {0+2+1+3}
        cu_tokens_across_dp_cpu = get_forward_context(
        ).dp_metadata.cu_tokens_across_dp_cpu

        hidden_states = self.naive_multicast(hidden_states,
                                             cu_tokens_across_dp_cpu)
        router_logits = self.naive_multicast(router_logits,
                                             cu_tokens_across_dp_cpu)

    # Matrix multiply.
    final_hidden_states = self.quant_method.apply(
        layer=self,
        x=hidden_states,
        router_logits=router_logits,
        top_k=self.top_k,
        renormalize=self.renormalize,
        use_grouped_topk=self.use_grouped_topk,
        global_num_experts=self.global_num_experts,
        expert_map=self.expert_map,
        topk_group=self.topk_group,
        num_expert_group=self.num_expert_group,
        custom_routing_function=self.custom_routing_function,
        scoring_func=self.scoring_func,
        e_score_correction_bias=self.e_score_correction_bias,
        activation=self.activation,
        apply_router_weight_on_input=self.apply_router_weight_on_input,
    )

    if self.dp_size > 1:
        # output all_reduce, all DPs broadcast each hidden_states & router_logits into dp group
        # within expert, dp_group all_reduce
        # ouf of expert, tp_group all_reduce
        all_hidden_states = get_dp_group().all_reduce(final_hidden_states)

        hidden_shape_dp = (self.dp_size, -1, org_hidden_shape[-1])
        final_hidden_states = all_hidden_states.reshape(hidden_shape_dp)
        final_hidden_states = final_hidden_states[self.dp_rank]
        final_hidden_states = final_hidden_states.reshape(org_hidden_shape)

    return final_hidden_states


def fused_moe_naive_multicast_rbln(self, x: torch.Tensor,
                        cu_tokens_across_dp_cpu: torch.Tensor):
    # as-is : [num_tokens, hidden_size]
    # to-be : buffer = [data_parallel_size*batch, seq, hidden_size], entire buffer for broadcast
    #         hidden = [batch, seq, hidden_size]
    # x.shape = [1, seq, hidden_size]
    # assert len(x.shape) == 3

    dp_size = self.dp_size
    # assert dp_size == cu_tokens_across_dp_cpu.size(-1)
    dp_rank = self.dp_rank
    batch_size = x.size(0)
    seq_size = x.size(1)
    hidden_size = x.size(2)

    if False:
        # each DP rank gather all inputs via torch.distributed.broadcast
        buffer = torch.empty((dp_size * batch_size, seq_size, hidden_size),
                    device=x.device,
                    dtype=x.dtype)
        # buffer[dp_rank] = x
        buffer = buffer.slice_scatter(x, dim=0, start=dp_rank, end=dp_rank+1)
        # gather all tensors of all ranks within dp group
        for rank in range(get_dp_group().world_size):
            get_dp_group().broadcast(buffer[rank:rank+1,:,:], rank)
        return buffer
    else:
        # each DP rank gather all inputs via torch.distributed.all_reduce
        # broadcast(value) == all_reduce(value for me or zeros for others)
        all_gather_buffer = None
        #buffer = torch.zeros((batch_size, seq_size, hidden_size),
        #            device=x.device,
        #            dtype=x.dtype)
        zeros = x - x
        for rank in range(get_dp_group().world_size):
            if rank == dp_rank:
                broadcast_tensor = get_dp_group().all_reduce(x)
            else:
                broadcast_tensor = get_dp_group().all_reduce(zeros)
            if all_gather_buffer is None:
                all_gather_buffer = broadcast_tensor
            else:
                all_gather_buffer = torch.cat((all_gather_buffer, broadcast_tensor), dim=0)
        return all_gather_buffer


FusedMoE.forward_oot = fused_moe_forward_rbln
if not envs.RBLN_MOE_CUSTOM_KERNEL:
    logger.info("[RBLN] fused moe, pytorch native kernel")
    UnquantizedFusedMoEMethod.forward_oot = unquantized_fused_moe_method_forward_rbln_rsd
else:
    logger.info("[RBLN] fused moe, RBLN custom kernel")
    UnquantizedFusedMoEMethod.forward_oot = unquantized_fused_moe_method_forward_rbln_custom
FusedMoE.naive_multicast = fused_moe_naive_multicast_rbln
