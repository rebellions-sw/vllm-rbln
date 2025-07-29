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
from typing import Any, Dict, Optional, cast

import torch
from vllm.config import VllmConfig
from vllm.logger import init_logger

from .base import ModelInputForRBLN
from .model_base import (RBLNOptimumDecoderMixin, RBLNOptimumDictTableMixin,
                         RBLNOptimumModelBase)

logger = init_logger(__name__)


class RBLNOptimumWhisperForConditionalGeneration(RBLNOptimumModelBase,
                                                 RBLNOptimumDecoderMixin,
                                                 RBLNOptimumDictTableMixin):
    INVALID_TOKEN = 100

    def __init__(
        self,
        vllm_config: VllmConfig,
    ) -> None:
        super().__init__(vllm_config=vllm_config)
        self.setup_decoder_mixin(
            attn_impl=self.attn_impl,
            padding_value=self.padding_value,
            vocab_size=self.model_config.get_vocab_size,
            use_multiple_decoder=False,
            default_batch_size=self.scheduler_config.max_num_seqs,
            decoder_batch_sizes=[self.batch_size],
        )
        self.dec_max_seq_len = self.model_config.max_model_len
        self.dec_lengths = [0] * self.batch_size
        self.table_mapping: Dict[str, int] = {}

    def get_table_id(
        self,
        is_prompt: bool,
        finished_requests_ids: list[str],
        running_requests_ids: list[str],
    ) -> list[int]:
        table_ids = self.get_table_mapping_values(
            self.table_mapping,
            self.decoder_batch_size,
            is_prompt,
            finished_requests_ids,
            running_requests_ids,
        )
        return cast(list[int], table_ids)

    def forward(self, model_input: ModelInputForRBLN,
                **kwargs) -> torch.Tensor:
        input_ids = model_input.input_tokens
        is_prompt = model_input.sampling_metadata.num_prompts > 0
        block_tables = model_input.block_tables

        finished_requests_ids = model_input.finished_requests_ids
        running_requests_ids = model_input.running_requests_ids
        request_nums = input_ids.shape[0]

        table_ids = self.get_table_id(is_prompt, finished_requests_ids,
                                      running_requests_ids)
        valid_block_ids = torch.tensor(table_ids)

        if is_prompt:
            if model_input.multi_modal_kwargs:
                input_features = self._parse_and_validate_audio_input(
                    **model_input.multi_modal_kwargs)
            if input_features is None:
                raise ValueError(
                    "Whisper requires `input_features` as an input.")

        cache_position = torch.zeros(request_nums, 1, dtype=torch.int32)

        kwargs = self.preprocess_for_decoder(is_prompt,
                                             block_tables,
                                             input_ids,
                                             cache_position,
                                             input_block_ids=valid_block_ids)
        input_ids = kwargs.pop("input_ids")
        cache_position = kwargs.pop("cache_position")
        block_tables = kwargs.pop("block_tables")

        if is_prompt:
            _ = self.model.encoder(input_features=input_features,
                                   block_tables=block_tables)
            lm_logits = torch.zeros(
                1, 1, self.model.config.vocab_size + self.INVALID_TOKEN)
            # Set the probability of INVALID_TOKEN (the last token in
            # the logits tensor) to 1.0.
            lm_logits[0][0][-1] = 1
            self.table_mapping[running_requests_ids[0]] = table_ids[0]
            self.dec_lengths[table_ids[0]] = 0

        else:
            input_ids[input_ids == (
                self.model.config.vocab_size + self.INVALID_TOKEN -
                1)] = self.model.config.decoder_start_token_id

            # FIXME Is it ok generate torch.zero tensor for each forward?
            # OR just generate pooled tensor in the model instance?
            decoder_attention_mask = torch.zeros(self.batch_size,
                                                 self.dec_max_seq_len,
                                                 dtype=torch.float32)
            # Generate cache_position using dec_lengths
            for batch_idx in valid_block_ids:
                cache_position[batch_idx] = self.dec_lengths[batch_idx]
                decoder_attention_mask[batch_idx, :cache_position[batch_idx] +
                                       1] = 1
                self.dec_lengths[batch_idx] += 1

            decoder_output = self.model.decoder(
                decoder_input_ids=input_ids.contiguous(),
                decoder_attention_mask=decoder_attention_mask,
                cache_position=cache_position,
                block_tables=block_tables,
            )

            lm_logits = decoder_output.logits
            lm_logits = lm_logits[valid_block_ids]
        return lm_logits

    def _parse_and_validate_audio_input(
            self, **kwargs: Any) -> Optional[torch.Tensor]:
        input_features = kwargs.pop("input_features", None)
        if input_features is not None:
            input_features = input_features.squeeze(0)
        return input_features

    def clear_dict_table(self):
        self.table_mapping.clear()
