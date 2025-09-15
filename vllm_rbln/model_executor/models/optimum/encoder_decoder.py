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
from typing import List, Optional, Union

import torch
from vllm.config import VllmConfig
from vllm.logger import init_logger

from .base import ModelInputForRBLN, version_error
from .model_base import RBLNOptimumDecoderMixin, RBLNOptimumModelBase

logger = init_logger(__name__)


class RBLNOptimumEncoderDecoder(RBLNOptimumModelBase, RBLNOptimumDecoderMixin):
    INVALID_TOKEN = 100

    def __init__(
        self,
        vllm_config: VllmConfig,
    ) -> None:
        super().__init__(vllm_config=vllm_config)
        # encoder length used for encoder_decoder architecture
        self.enc_lengths = [0] * self.batch_size
        self.setup_decoder_mixin(
            attn_impl=self.attn_impl,
            vocab_size=self.model_config.get_vocab_size,
            use_multiple_decoder=False,
            default_batch_size=self.scheduler_config.max_num_seqs,
            decoder_batch_sizes=[self.batch_size],
        )

    def _forward(
        self,
        enc_lengths: List[int],  # current attention_mask length
        input_ids: torch.LongTensor = None,
        cache_position: Union[List[torch.Tensor], torch.Tensor] = None,
        batch_idx: Optional[torch.LongTensor] = None,
        block_tables: torch.Tensor = None,
        **kwargs,
    ):
        # When using vLLM, the output of the encoder needs to include
        # an additional token (e.g., vocab_size + INVALID_TOKEN).
        # This value serves as the start_token_id in the decoder.
        # The decoder will then use (vocab_size + INVALID_TOKEN - 1)
        # as the actual start_token_id.

        # Encoder
        if batch_idx is not None:
            enc_attention_mask = torch.zeros(
                1, self.model.rbln_config.enc_max_seq_len, dtype=torch.float32)
            enc_attention_mask[0][:enc_lengths[batch_idx] + 1] = 1

            padding_need = (self.model.rbln_config.enc_max_seq_len -
                            input_ids.shape[-1])
            input_ids = torch.nn.functional.pad(input_ids, (0, padding_need))

            _ = self.model.encoder(input_ids,
                                   enc_attention_mask,
                                   block_tables=block_tables)

            logits = torch.zeros(
                1, 1, self.model.config.vocab_size + self.INVALID_TOKEN)
            # Set the probability of INVALID_TOKEN (the last token in
            # the logits tensor) to 1.0.
            logits[0][0][-1] = 1

        # Decoder
        else:
            # Replace INVALID_TOKEN markers with the decoder start token ID
            input_ids[input_ids == (
                self.model.config.vocab_size + self.INVALID_TOKEN -
                1)] = self.model.config.decoder_start_token_id
            cache_position[cache_position !=
                           0] = cache_position[cache_position != 0] - 2

            enc_attention_mask = torch.zeros(
                self.model.rbln_config.batch_size,
                self.model.rbln_config.enc_max_seq_len,
                dtype=torch.float32,
            )
            dec_attention_mask = torch.zeros(
                self.model.rbln_config.batch_size,
                self.model.rbln_config.dec_max_seq_len,
                dtype=torch.float32,
            )

            for batch_idx in range(self.model.rbln_config.batch_size):
                enc_attention_mask[batch_idx, :enc_lengths[batch_idx] + 1] = 1

            if self.model.decoder is None:
                raise version_error

            logits = self.model.decoder(
                decoder_input_ids=input_ids,
                attention_mask=enc_attention_mask,
                decoder_attention_mask=dec_attention_mask,
                cache_position=cache_position,
                block_tables=block_tables,
            ).logits

        return logits

    def forward(self, model_input: ModelInputForRBLN,
                **kwargs) -> torch.Tensor:
        input_ids = model_input.input_tokens
        cache_position = model_input.input_positions
        is_prompt = model_input.sampling_metadata.num_prompts > 0
        block_tables = model_input.block_tables
        valid_block_ids = [
            block_table[0].item() for block_table in block_tables
        ]
        batch_idx = block_tables[0][0] if is_prompt else None

        kwargs = self.preprocess_for_decoder(is_prompt,
                                             block_tables,
                                             self.kv_block_adapter,
                                             input_ids,
                                             cache_position,
                                             input_block_ids=valid_block_ids)
        input_ids = kwargs.pop("input_ids")
        cache_position = kwargs.pop("cache_position")
        block_tables = kwargs.pop("block_tables")

        # NOTE multi-batch-size is not supported in encoder-decoder?
        if is_prompt:
            # prefill batch_size is always 1
            assert cache_position.shape[0] == 1
            self.enc_lengths[batch_idx] = cache_position[0][-1].item()

        logits = self._forward(
            input_ids=input_ids,
            cache_position=cache_position,
            batch_idx=batch_idx,
            enc_lengths=self.enc_lengths,
            block_tables=block_tables,
        )

        if not is_prompt:
            logits = logits[valid_block_ids]

        return logits
