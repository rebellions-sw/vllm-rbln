# Copyright 2026 Rebellions Inc. All rights reserved.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at:

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import itertools
from collections.abc import Sequence

import torch
from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.v1.sample.logits_processor import (
    STR_POOLING_REJECTS_LOGITSPROCS,
    STR_SPEC_DEC_REJECTS_LOGITSPROCS,
    LogitsProcessors,
    _load_custom_logitsprocs,
)
from vllm.v1.sample.logits_processor.builtin import (
    LogitBiasLogitsProcessor,
    MinPLogitsProcessor,
    MinTokensLogitsProcessor,
)
from vllm.v1.sample.logits_processor.interface import LogitsProcessor

logger = init_logger(__name__)


class RBLNMinTokensLogitsProcessor(MinTokensLogitsProcessor):
    # index_put_ requires the value dtype and device to exactly match the
    # logits, and two kinds of logits reach one instance: a step without drafts
    # samples model-dtype logits on the device through the RBLN sampler, while
    # a speculative step samples on the host (see RBLNModelRunner._sample) --
    # float32-upcast target logits in apply_with_spec_decode() and the bonus
    # logits in apply(). The -inf constant is therefore synced to the incoming
    # logits per call, with one cached tensor per (dtype, device), and the
    # index tensors are put on the logits device as well.
    neg_inf_tensor: torch.Tensor
    device: torch.device

    def __init__(
        self, vllm_config: VllmConfig, device: torch.device, is_pin_memory: bool
    ):
        super().__init__(vllm_config, device, is_pin_memory)
        self._neg_inf_tensors = {
            (self.neg_inf_tensor.dtype, self.neg_inf_tensor.device): self.neg_inf_tensor
        }

    def _sync_neg_inf(self, logits: torch.Tensor):
        key = (logits.dtype, logits.device)
        tensor = self._neg_inf_tensors.get(key)
        if tensor is None:
            # Built fresh rather than copied across from the original, which
            # may sit on the other device.
            tensor = self._neg_inf_tensors[key] = torch.tensor(
                -float("inf"), dtype=logits.dtype, device=logits.device
            )
        self.neg_inf_tensor = tensor

    def apply(self, logits: torch.Tensor) -> torch.Tensor:
        if not self.min_toks:
            return logits
        self._sync_neg_inf(logits)
        rows, toks = self.logits_slice
        if rows.device != logits.device:
            # update_state() built the slice on `self.device`; a speculative
            # step's bonus logits are on the host.
            logits.index_put_(
                (rows.to(logits.device), toks.to(logits.device)), self.neg_inf_tensor
            )
            return logits
        return super().apply(logits)

    def apply_with_spec_decode(
        self, logits: torch.Tensor, num_draft_tokens: list[int]
    ) -> torch.Tensor:
        if not self.min_toks:
            return logits
        self._sync_neg_inf(logits)
        # Upstream allocates the row / token index tensors on `self.device`,
        # which is where update_state() keeps the non-spec `logits_slice`; the
        # spec-decode logits may live elsewhere, so point it at them for the call.
        device = self.device
        self.device = logits.device
        try:
            return super().apply_with_spec_decode(logits, num_draft_tokens)
        finally:
            self.device = device


class RBLNLogitBiasLogitsProcessor(LogitBiasLogitsProcessor):
    # bias_tensor is rebuilt as float32 on every state change, so a one-time
    # cast would not stick: it is synced to the incoming logits dtype on each
    # apply() call.
    bias_tensor: torch.Tensor

    def apply(self, logits: torch.Tensor) -> torch.Tensor:
        if self.biases and self.bias_tensor.dtype != logits.dtype:
            self.bias_tensor = self.bias_tensor.to(logits.dtype)
        return super().apply(logits)


class RBLNMinPLogitsProcessor(MinPLogitsProcessor):
    # min_p is re-sliced from the float32 buffer on state changes and
    # multiplied in place into model-dtype probabilities, so it is synced
    # to the incoming logits dtype on each apply() call.
    min_p: torch.Tensor

    def apply(self, logits: torch.Tensor) -> torch.Tensor:
        if self.min_p_count and self.min_p.dtype != logits.dtype:
            self.min_p = self.min_p.to(logits.dtype)
        return super().apply(logits)


RBLN_BUILTIN_LOGITS_PROCESSORS: list[type[LogitsProcessor]] = [
    RBLNMinTokensLogitsProcessor,
    RBLNLogitBiasLogitsProcessor,
    RBLNMinPLogitsProcessor,
]


def build_rbln_logitsprocs(
    vllm_config: VllmConfig,
    device: torch.device,
    is_pin_memory: bool,
    is_pooling_model: bool,
    custom_logitsprocs: Sequence[str | type[LogitsProcessor]] = (),
) -> LogitsProcessors:
    """Mirror vLLM's build_logitsprocs with dtype-aware builtin processors."""
    if is_pooling_model:
        if custom_logitsprocs:
            raise ValueError(STR_POOLING_REJECTS_LOGITSPROCS)
        return LogitsProcessors()

    if vllm_config.speculative_config:
        if custom_logitsprocs:
            raise ValueError(STR_SPEC_DEC_REJECTS_LOGITSPROCS)
        logger.warning(
            "min_p and logit_bias parameters won't work with speculative decoding."
        )
        return LogitsProcessors(
            [RBLNMinTokensLogitsProcessor(vllm_config, device, is_pin_memory)]
        )

    custom_logitsprocs_classes = _load_custom_logitsprocs(custom_logitsprocs)
    return LogitsProcessors(
        ctor(vllm_config, device, is_pin_memory)
        for ctor in itertools.chain(
            RBLN_BUILTIN_LOGITS_PROCESSORS, custom_logitsprocs_classes
        )
    )
