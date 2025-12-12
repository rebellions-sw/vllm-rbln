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
from typing import Any, Optional

import torch
import vllm.envs as envs
from transformers import AutoTokenizer
from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.model_executor.models.gemma3_mm import (Gemma3DummyInputsBuilder,
                                                  Gemma3ImageInputs,
                                                  Gemma3ImagePixelInputs,
                                                  Gemma3MultiModalProcessor,
                                                  Gemma3ProcessingInfo)
from vllm.model_executor.models.interfaces import SupportsMultiModal
from vllm.model_executor.models.interfaces_base import (
    VllmModelForTextGeneration)
from vllm.model_executor.models.utils import flatten_bn
from vllm.multimodal import MULTIMODAL_REGISTRY

from .base import ModelInputForRBLN, version_error
from .model_base import RBLNOptimumDecoderMixin, RBLNOptimumModelBase
from .optimum_attention import (HybridAttentionManager,
                                HybridAttentionStrategy)

class RBLNOptimumHybridAttentionForCausalLM(
        RBLNOptimumModelBase,
        RBLNOptimumDecoderMixin,
        VllmModelForTextGeneration,
):

    def __init__(
        self,
        vllm_config: VllmConfig,
    ) -> None:
        super().__init__(vllm_config=vllm_config)
        self.setup_decoder_mixin(
            attn_impl=self.attn_impl,
            vocab_size=self.model_config.get_vocab_size,
            use_multiple_decoder=getattr(
                self.model.rbln_config,
                "use_multiple_decoder",
                False,
            ),
            default_batch_size=self.scheduler_config.max_num_seqs,
            decoder_batch_sizes=self.model.rbln_config.decoder_batch_sizes,
            num_blocks=self.kv_block_adapter._estimated_num_blocks(),
        )

        # FIXME Loading tokenizer in model runner is a temporary solution.
        tokenizer = AutoTokenizer.from_pretrained(self.model_config.tokenizer)

        self.strategy = HybridAttentionStrategy()
        self.attention_manager: HybridAttentionManager \
            = HybridAttentionManager(self.strategy)

    def forward(self, model_input: ModelInputForRBLN,
                **kwargs) -> torch.Tensor:
        input_ids = model_input.input_tokens
        position_ids = model_input.input_positions
        block_tables = model_input.block_tables

        if envs.VLLM_USE_V1:
            is_prompt = model_input.is_prompt
        else:
            is_prompt = model_input.sampling_metadata.num_prompts > 0

        finished_requests_ids = model_input.finished_requests_ids
        running_requests_ids = model_input.running_requests_ids
        request_nums = input_ids.shape[0]

        # In prefill phase, the length of list must be 1
        sliding_window_table_ids = \
        self.attention_manager.get(
                is_prompt,
                self.decoder_batch_size,
                running_requests_ids,
                finished_requests_ids,
                input_ids=input_ids,
            )

        kwargs = self.preprocess_for_decoder(is_prompt, block_tables,
                                             input_ids, position_ids)

        # [prefill] the length of the padded cache is calculated
        # during the forward pass and stored in self.sliding_window_table.
        # [decode] `cache_position` and `position_ids` are distinguished
        # due to the padding space reserved for the sliding window.
        cache_position = kwargs.pop("cache_position")
        input_ids = kwargs.pop("input_ids")
        block_tables = kwargs.pop("block_tables")

        if is_prompt:
            # inputs_embeds = None
            prefill_batch_idx = sliding_window_table_ids[0]
            local_block_table_id = torch.tensor([prefill_batch_idx],
                                                dtype=torch.int16)

            # pixel_values = self.get_pixel_values(model_input)
            # inputs_embeds = self.model._preprocess_prefill(
            #     input_ids, inputs_embeds, pixel_values)
            if self.model.prefill_decoder is None:
                raise version_error
            # attention_mask = attention_masks[0]
            output = self.model.prefill_decoder(
                input_ids=input_ids,
                cache_position=cache_position,
                # attention_mask=attention_mask,
                local_block_tables=local_block_table_id,
                block_tables=block_tables,
            )
            logits = output.logits
            # updated_attention_mask = output.attention_mask
            # updated_padded_cache_length = output.padded_cache_lengths

            assert len(running_requests_ids) == 1
            self.attention_manager.add(
                running_requests_id=running_requests_ids[0],
                local_table_id=sliding_window_table_ids[0],
                # pad_len=updated_padded_cache_length,
                # attention_mask=updated_attention_mask,
            )
        else:
            if self.model.decoders is None:
                raise ValueError("Decoders is None")
            padded_batch_size = kwargs.pop("padded_batch_size",
                                           self.decoder_batch_size)
            self.model.decoder = (
                self.model.decoders[padded_batch_size])
            # (
            #     local_block_table_id,
            #     cache_position,
            #     position_ids,
            #     attention_mask,
            # ) = self.attention_manager.preprocess(
            #     sliding_window_table_ids,
            #     cache_position,
            #     request_nums,
            #     padded_batch_size,
            #     pad_lens=padded_cache_lengths,
            #     attention_masks=attention_masks,
            # )
            local_block_table_id, cache_position = \
                self.attention_manager.preprocess(
                sliding_window_table_ids,
                cache_position,
                request_nums,
                padded_batch_size,
            )

            # attention_mask = self.attention_manager.update(
            #     running_requests_ids,
            #     attention_mask,
            #     cache_position,
            # )

            logits = self.model.decoder(
                input_ids=input_ids,
                cache_position=cache_position,
                block_tables=block_tables,
                local_block_tables=local_block_table_id,
                # attention_mask=attention_mask,
                # position_ids=position_ids,
            ).logits

        if not is_prompt:
            logits = logits[:request_nums]
        return logits