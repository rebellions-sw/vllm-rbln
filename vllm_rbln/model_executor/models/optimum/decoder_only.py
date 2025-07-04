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
from vllm.config import ModelConfig, SchedulerConfig
from vllm.logger import init_logger

from .base import ModelInputForRBLN, version_error
from .model_base import RBLNOptimumDecoderMixin, RBLNOptimumModelBase

logger = init_logger(__name__)


class RBLNOptimumForCausalLM(RBLNOptimumModelBase, RBLNOptimumDecoderMixin):

    def __init__(
        self,
        model_config: ModelConfig,
        scheduler_config: SchedulerConfig,
        **kwargs,
    ) -> None:
        super().__init__(
            model_config=model_config,
            scheduler_config=scheduler_config,
        )
        self.setup_decoder_mixin(
            attn_impl=self.attn_impl,
            padding_value=self.padding_value,
            vocab_size=model_config.get_vocab_size,
            use_multiple_decoder=getattr(self.model.rbln_config,
                                         "use_multiple_decoder", False),
            default_batch_size=self.scheduler_config.max_num_seqs,
            decoder_batch_sizes=self.model.rbln_config.decoder_batch_sizes,
        )

    def forward(self, model_input: ModelInputForRBLN) -> torch.Tensor:
        input_ids = model_input.input_tokens
        cache_position = model_input.input_positions
        block_tables = model_input.block_tables
        is_prompt = model_input.sampling_metadata.num_prompts > 0
        request_nums = input_ids.shape[0]
        kwargs = self.preprocess_for_decoder(
            is_prompt,
            block_tables,
            input_ids,
            cache_position,
        )
        padded_batch_size = kwargs.pop("padded_batch_size",
                                       self.decoder_batch_size)

        if is_prompt:
            if self.model.prefill_decoder is None:
                raise version_error

            return self.model.prefill_decoder(**kwargs).logits
        else:
            self.model.decoder = self.model.decoders[padded_batch_size]

            logits = self.model.decoder(**kwargs).logits
            if self.attn_impl != "flash_attn":
                return logits[:request_nums]

            return logits[:model_input.block_tables.shape[0]]
