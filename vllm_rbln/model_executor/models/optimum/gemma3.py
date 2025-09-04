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
import vllm.envs as env
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
from vllm.multimodal import MULTIMODAL_REGISTRY

from .base import ModelInputForRBLN, version_error
from .model_base import RBLNOptimumDecoderMixin, RBLNOptimumModelBase
from .optimum_attention import (HybridAttentionImageManager,
                                HybridAttentionImageStrategy)

logger = init_logger(__name__)


class RBLNGemma3MultiModalProcessor(Gemma3MultiModalProcessor):

    def _pad_for_gemma3(self, prompt_ids: list[int], prompt: str):
        token_type_ids = (torch.tensor(prompt_ids) ==
                          self.info.get_hf_processor().image_token_id)

        image_prefill_chunk_size = self.info.get_hf_processor(
        ).image_seq_length
        # Find image start positions
        image_starts = [
            s for s in torch.where(token_type_ids)[0]
            if torch.all(token_type_ids[s:s + image_prefill_chunk_size])
        ]
        padded_seq_len = 0
        for image_start in image_starts:
            pad_needed = (
                image_prefill_chunk_size -
                (image_start + padded_seq_len) % image_prefill_chunk_size)
            padded_seq_len += pad_needed

        pad_token = self.info.get_hf_processor().tokenizer.pad_token
        pad_token_id = self.info.get_hf_processor().tokenizer.pad_token_id

        prompt_ids = prompt_ids + [pad_token_id] * padded_seq_len
        prompt = prompt + pad_token * padded_seq_len
        return prompt_ids, prompt

    def apply(self, *args, **kwargs):
        output = super().apply(*args, **kwargs)
        prompt_ids, prompt = self._pad_for_gemma3(output["prompt_token_ids"],
                                                  output["prompt"])

        output["prompt_token_ids"] = prompt_ids
        output["prompt"] = prompt

        return output


@MULTIMODAL_REGISTRY.register_processor(
    RBLNGemma3MultiModalProcessor,
    info=Gemma3ProcessingInfo,
    dummy_inputs=Gemma3DummyInputsBuilder,
)
class RBLNOptimumGemma3ForConditionalGeneration(
        RBLNOptimumModelBase,
        RBLNOptimumDecoderMixin,
        VllmModelForTextGeneration,
        SupportsMultiModal,
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
                self.model.rbln_config.language_model,
                "use_multiple_decoder",
                False,
            ),
            default_batch_size=self.scheduler_config.max_num_seqs,
            decoder_batch_sizes=self.model.rbln_config.language_model.
            decoder_batch_sizes,
        )

        # FIXME Loading tokenizer in model runner is a temporary solution.
        tokenizer = AutoTokenizer.from_pretrained(self.model_config.tokenizer)

        self.strategy = HybridAttentionImageStrategy(tokenizer.pad_token_id)
        self.attention_manager: HybridAttentionImageManager \
            = HybridAttentionImageManager(self.strategy)

    def forward(self, model_input: ModelInputForRBLN,
                **kwargs) -> torch.Tensor:
        input_ids = model_input.input_tokens
        position_ids = model_input.input_positions
        block_tables = model_input.block_tables

        if env.VLLM_USE_V1:
            is_prompt = model_input.is_prompt
        else:
            is_prompt = model_input.sampling_metadata.num_prompts > 0

        finished_requests_ids = model_input.finished_requests_ids
        running_requests_ids = model_input.running_requests_ids
        request_nums = input_ids.shape[0]

        # In prefill phase, the length of list must be 1
        sliding_window_table_ids, padded_cache_lengths, attention_masks = \
        self.attention_manager.get(
                is_prompt,
                self.decoder_batch_size,
                running_requests_ids,
                finished_requests_ids,
                input_ids=input_ids,
            )

        kwargs = self.preprocess_for_decoder(is_prompt, block_tables,
                                             self.kv_block_adapter, input_ids,
                                             position_ids)

        # [prefill] the length of the padded cache is calculated
        # during the forward pass and stored in self.sliding_window_table.
        # [decode] `cache_position` and `position_ids` are distinguished
        # due to the padding space reserved for the sliding window.
        cache_position = kwargs.pop("cache_position")
        input_ids = kwargs.pop("input_ids")
        block_tables = kwargs.pop("block_tables")

        if is_prompt:
            inputs_embeds = None
            prefill_batch_idx = sliding_window_table_ids[0]
            local_block_table_id = torch.tensor([prefill_batch_idx],
                                                dtype=torch.int16)
            # token_type_ids model_input != token_type_ids of gemma3
            # https://github.com/huggingface/transformers/blob/d0c9c66d1c09df3cd70bf036e813d88337b20d4c/src/transformers/models/gemma3/processing_gemma3.py#L143
            token_type_ids = torch.zeros_like(input_ids)
            token_type_ids[input_ids ==
                           self.model.config.image_token_index] = 1

            pixel_values = self.get_pixel_values(model_input)
            inputs_embeds = self.model._preprocess_prefill(
                input_ids, inputs_embeds, pixel_values)
            if self.model.language_model.prefill_decoder is None:
                raise version_error
            assert attention_masks is not None
            attention_mask = attention_masks[0]
            output = self.model.language_model.prefill_decoder(
                inputs_embeds=inputs_embeds,
                cache_position=cache_position,
                attention_mask=attention_mask,
                local_block_tables=local_block_table_id,
                block_tables=block_tables,
                token_type_ids=token_type_ids,
            )
            logits = output.logits
            updated_attention_mask = output.attention_mask
            updated_padded_cache_length = output.padded_cache_lengths

            assert len(running_requests_ids) == 1
            self.attention_manager.add(
                running_requests_id=running_requests_ids[0],
                local_table_id=sliding_window_table_ids[0],
                pad_len=updated_padded_cache_length,
                attention_mask=updated_attention_mask,
            )
        else:
            if self.model.language_model.decoders is None:
                raise ValueError("Decoders is None")
            padded_batch_size = kwargs.pop("padded_batch_size",
                                           self.decoder_batch_size)
            self.model.language_model.decoder = (
                self.model.language_model.decoders[padded_batch_size])
            (
                local_block_table_id,
                cache_position,
                position_ids,
                attention_mask,
            ) = self.attention_manager.preprocess(
                sliding_window_table_ids,
                cache_position,
                request_nums,
                padded_batch_size,
                pad_lens=padded_cache_lengths,
                attention_masks=attention_masks,
            )

            attention_mask = self.attention_manager.update(
                running_requests_ids,
                attention_mask,
                cache_position,
            )

            logits = self.model.language_model.decoder(
                input_ids=input_ids,
                cache_position=cache_position,
                block_tables=block_tables,
                local_block_tables=local_block_table_id,
                attention_mask=attention_mask,
                position_ids=position_ids,
            ).logits

        if not is_prompt:
            logits = logits[:request_nums]
        return logits

    def get_pixel_values(self, model_input: ModelInputForRBLN):
        image_input = None

        if model_input.multi_modal_kwargs:
            image_input = self._parse_and_validate_image_input(
                **model_input.multi_modal_kwargs)
            if image_input is not None:
                assert image_input["type"] == "pixel_values"
                pixel_values = image_input["pixel_values"]

        else:
            pixel_values = None

        return pixel_values

    def _parse_and_validate_image_input(
            self, **kwargs: Any) -> Optional[Gemma3ImageInputs]:
        pixel_values: torch.Tensor = kwargs.get("pixel_values")
        num_crops: torch.Tensor = kwargs.get("num_crops")
        embed_is_patch = kwargs.get("embed_is_patch")
        num_embeds = kwargs.get("num_embeds")

        if pixel_values is None:
            return None

        if not isinstance(pixel_values, (torch.Tensor, list)):
            raise ValueError("Incorrect type of pixel values. "
                             f"Got type: {type(pixel_values)}")

        if env.VLLM_USE_V1:
            pixel_values = pixel_values.squeeze(1)
        else:
            pixel_values = pixel_values.squeeze(0)

        return Gemma3ImagePixelInputs(
            type="pixel_values",
            pixel_values=self._validate_pixel_values(pixel_values),
            num_patches=num_crops + 1,
            embed_is_patch=embed_is_patch,
            num_embeds=num_embeds,
        )

    def _validate_pixel_values(self, data: torch.Tensor) -> torch.Tensor:
        h = w = self.model.config.vision_config.image_size
        expected_dims = (3, h, w)

        def _validate_shape(d: torch.Tensor):
            actual_dims = tuple(d.shape)

            if actual_dims != expected_dims:
                expected_expr = str(expected_dims)
                raise ValueError(
                    "The expected shape of pixel values per image per batch "
                    f" per patch is {expected_expr}. "
                    f"You supplied {tuple(d.shape)}.")

        for d in data:
            _validate_shape(d)

        return data
