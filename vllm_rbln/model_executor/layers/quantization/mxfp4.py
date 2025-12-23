from typing import Callable, Optional, Union

import torch
import vllm.model_executor.layers.quantization.mxfp4 as upstream
from torch import Tensor
from vllm.model_executor.layers.fused_moe import (FusedMoE, FusedMoEConfig,
                                                  FusedMoEMethodBase)
from vllm.model_executor.layers.fused_moe import modular_kernel as mk
from vllm.model_executor.layers.fused_moe.config import FusedMoEQuantConfig
from vllm.model_executor.utils import set_weight_attrs

from vllm_rbln.logger import init_logger

logger = init_logger(__name__)


# kernel for gpt_oss, with built-in swigluoai activation
@torch.library.custom_op(
    "rbln_custom_ops::custom_moe_glu_mxfp4",
    mutates_args=(),
)
def custom_moe_glu_mxfp4(
    hidden_states: Tensor,
    gate_proj_blocks: Tensor,
    gate_proj_scales: Tensor,
    gate_proj_bias: Tensor,
    up_proj_blocks: Tensor,
    up_proj_scales: Tensor,
    up_proj_bias: Tensor,
    down_proj_blocks: Tensor,
    down_proj_scales: Tensor,
    down_proj_bias: Tensor,
    router_logits: Tensor,
    alpha: Tensor,
    limit: Tensor,
    k: int,
    post_norm: bool = True,
) -> Tensor:
    """
    Customized MoE GLU operation.

    Expected tensor shapes:
    - hidden_states: [batch*seq_len, hidden_size]
    - gate_proj_blocks: [num_experts, intermediate_size, hidden_size // 2]
    - gate_proj_scales: [num_experts, intermediate_size, hidden_size // 32]
    - gate_proj_bias: [num_experts, intermediate_size]
    - up_proj_blocks: [num_experts, intermediate_size, hidden_size // 2]
    - up_proj_scales: [num_experts, intermediate_size, hidden_size // 32]
    - up_proj_bias: [num_experts, intermediate_size]
    - down_proj_blocks: [num_experts, hidden_size, intermediate_size // 2]
    - down_proj_scales: [num_experts, hidden_size, intermediate_size // 32]
    - masked_routing_weight: [batch * seq_len, num_experts]
    - expert_select_count: [num_experts]
    - alpha: []
    - limit: []

    Returns:
        Tensor: [batch * seq_len, hidden_size]
    """

    return torch.empty_like(hidden_states)


@custom_moe_glu_mxfp4.register_fake
def custom_moe_glu_mxfp4_fake(
    hidden_states: Tensor,
    gate_proj_blocks: Tensor,
    gate_proj_scales: Tensor,
    gate_proj_bias: Tensor,
    up_proj_blocks: Tensor,
    up_proj_scales: Tensor,
    up_proj_bias: Tensor,
    down_proj_blocks: Tensor,
    down_proj_scales: Tensor,
    down_proj_bias: Tensor,
    router_logits: Tensor,
    alpha: Tensor,
    limit: Tensor,
    k: int,
    post_norm: bool = True,
) -> Tensor:
    return torch.empty_like(hidden_states)


class Mxfp4MoEMethod(FusedMoEMethodBase):

    def __init__(self, moe: FusedMoEConfig):
        super().__init__(moe)
        self.moe = moe

        self._cache_permute_indices: dict[torch.Size, torch.Tensor] = {}

    def create_weights(self, layer: torch.nn.Module, num_experts: int,
                       hidden_size: int, intermediate_size_per_partition: int,
                       params_dtype: torch.dtype, **extra_weight_attrs):
        assert isinstance(layer, FusedMoE)

        self.num_experts = num_experts
        weight_dtype = torch.uint8
        scale_dtype = torch.uint8

        mxfp4_block = 32

        intermediate_size_per_partition_after_pad = \
            intermediate_size_per_partition

        # NOTE: upstream rounds up intermediate_size_per_partition/hidden_size
        assert intermediate_size_per_partition % 64 == 0

        self.intermediate_size = intermediate_size_per_partition_after_pad
        self.hidden_size = hidden_size
        # Fused gate_up_proj (column parallel)
        w13_weight = torch.nn.Parameter(
            torch.zeros(
                num_experts,
                2 * intermediate_size_per_partition_after_pad,
                hidden_size // 2,
                dtype=weight_dtype,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w13_weight", w13_weight)
        set_weight_attrs(w13_weight, extra_weight_attrs)

        w13_weight_scale = torch.nn.Parameter(
            torch.zeros(
                num_experts,
                2 * intermediate_size_per_partition_after_pad,
                hidden_size // mxfp4_block,
                dtype=scale_dtype,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w13_weight_scale", w13_weight_scale)
        set_weight_attrs(w13_weight_scale, extra_weight_attrs)

        w13_bias = torch.nn.Parameter(
            torch.zeros(
                num_experts,
                2 * intermediate_size_per_partition_after_pad,
                dtype=torch.bfloat16,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w13_bias", w13_bias)
        set_weight_attrs(w13_bias, extra_weight_attrs)

        # down_proj (row parallel)
        w2_weight = torch.nn.Parameter(
            torch.zeros(
                num_experts,
                hidden_size,
                intermediate_size_per_partition_after_pad // 2,
                dtype=weight_dtype,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w2_weight", w2_weight)
        set_weight_attrs(w2_weight, extra_weight_attrs)

        w2_weight_scale = torch.nn.Parameter(
            torch.zeros(
                num_experts,
                hidden_size,
                intermediate_size_per_partition_after_pad // mxfp4_block,
                dtype=scale_dtype,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w2_weight_scale", w2_weight_scale)
        set_weight_attrs(w2_weight_scale, extra_weight_attrs)

        w2_bias = torch.nn.Parameter(
            torch.zeros(
                num_experts,
                hidden_size,
                dtype=torch.bfloat16,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w2_bias", w2_bias)
        set_weight_attrs(w2_bias, extra_weight_attrs)

    def process_weights_after_loading(self, layer):
        assert isinstance(layer, FusedMoE)

        # w1
        layer.register_buffer("gate_proj_blocks",
                              layer.w13_weight.data[:, ::2])
        layer.register_buffer("gate_proj_scales",
                              layer.w13_weight_scale.data[:, ::2])
        layer.register_buffer("gate_proj_bias", layer.w13_bias.data[:, ::2])

        # w3
        layer.register_buffer("up_proj_blocks", layer.w13_weight.data[:, 1::2])
        layer.register_buffer("up_proj_scales",
                              layer.w13_weight_scale.data[:, 1::2])
        layer.register_buffer("up_proj_bias", layer.w13_bias.data[:, 1::2])

        # w2
        layer.register_buffer("down_proj_blocks", layer.w2_weight.data)
        layer.register_buffer("down_proj_scales", layer.w2_weight_scale.data)
        layer.register_buffer("down_proj_bias", layer.w2_bias.data)

    def select_gemm_impl(
        self,
        prepare_finalize: mk.FusedMoEPrepareAndFinalize,
        layer: torch.nn.Module,
    ) -> mk.FusedMoEPermuteExpertsUnpermute:
        # NOTE(RBLN): this is used only for "modular kernel"
        raise NotImplementedError()

    def get_fused_moe_quant_config(
            self, layer: torch.nn.Module) -> FusedMoEQuantConfig | None:
        # NOTE(RBLN): this is used only for "modular kernel"
        raise NotImplementedError

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        router_logits: torch.Tensor,
        top_k: int,
        renormalize: bool,
        use_grouped_topk: bool = False,
        topk_group: Optional[int] = None,
        num_expert_group: Optional[int] = None,
        global_num_experts: int = -1,
        expert_map: Optional[torch.Tensor] = None,
        custom_routing_function: Optional[Callable] = None,
        scoring_func: str = "softmax",
        routed_scaling_factor: float = 1.0,
        e_score_correction_bias: Optional[torch.Tensor] = None,
        apply_router_weight_on_input: bool = False,
        activation: str = "silu",
        enable_eplb: bool = False,
        expert_load_view: Optional[torch.Tensor] = None,
        logical_to_physical_map: Optional[torch.Tensor] = None,
        logical_replica_count: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:

        orig_shape = x.shape
        x = x.view(-1, self.hidden_size)
        router_logits = router_logits.view(-1, self.num_experts)

        if activation == "swigluoai":
            # TODO: use expert_map
            output = torch.ops.rbln_custom_ops.custom_moe_glu_mxfp4(
                x,
                layer.gate_proj_blocks,
                layer.gate_proj_scales,
                layer.gate_proj_bias,
                layer.up_proj_blocks,
                layer.up_proj_scales,
                layer.up_proj_bias,
                layer.down_proj_blocks,
                layer.down_proj_scales,
                layer.down_proj_bias,
                router_logits,
                # alpha (hardcoded in vllm as well?)
                torch.tensor(1.702, dtype=x.dtype),
                # swiglu_limit
                torch.tensor(7.0, dtype=x.dtype),
                k=top_k,
            )
        else:
            raise NotImplementedError(activation)

        return output.view(orig_shape)


# We do this because upstream uses Mxfp4MoEMethod for all non-xpu platforms
# and it doesn't expose interface for OOT kernels.
upstream.Mxfp4MoEMethod = Mxfp4MoEMethod
