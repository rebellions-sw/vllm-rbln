from vllm_rbln.model_executor.models.optimum.base import ModelInputForRBLN
from vllm_rbln.model_executor.models.optimum.gemma3 import RBLNGemma3MultiModalProcessor, RBLNOptimumGemma3ForConditionalGeneration, Gemma3ProcessingInfo, Gemma3DummyInputsBuilder
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.config import VllmConfig
from typing import Any, Optional
from vllm.model_executor.models.utils import flatten_bn
from vllm.model_executor.models.paligemma import PaliGemmaImageInputs, PaliGemmaImagePixelInputs, PaliGemmaImageEmbeddingInputs

from .base import ModelInputForRBLN
from .model_base import RBLNOptimumDecoderMixin, RBLNOptimumModelBase
import torch
from vllm.logger import init_logger

import vllm.envs as envs
class RBLNOptimumPaliGemmaForConditionalGeneration(RBLNOptimumModelBase,
                                                  RBLNOptimumDecoderMixin):

    def __init__(
        self,
        vllm_config: VllmConfig,
    ) -> None:
        super().__init__(vllm_config=vllm_config)
        self.setup_decoder_mixin(
            attn_impl=self.attn_impl,
            vocab_size=self.model_config.get_vocab_size,
            use_multiple_decoder=getattr(self.model.rbln_config.language_model,
                                         "use_multiple_decoder", False),
            default_batch_size=self.scheduler_config.max_num_seqs,
            decoder_batch_sizes=self.model.rbln_config.language_model.
            decoder_batch_sizes,
            num_blocks=self.kv_block_adapter._estimated_num_blocks(),
        )

    def forward(self, model_input: ModelInputForRBLN,
                **kwargs) -> torch.Tensor:
        input_ids = model_input.input_tokens
        cache_position = model_input.input_positions
        block_tables = model_input.block_tables

        request_nums = input_ids.shape[0]
        if envs.VLLM_USE_V1:
            is_prompt = model_input.is_prompt
        else:
            is_prompt = model_input.sampling_metadata.num_prompts > 0

        kwargs = self.preprocess_for_decoder(is_prompt, block_tables,
                                             input_ids, cache_position)

        if is_prompt:
            if model_input.multi_modal_kwargs:
                pixel_values = self.get_pixel_values(model_input)
            else:
                pixel_values = None

            block_tables = kwargs.pop("block_tables")
            input_ids = kwargs.pop("input_ids")
            cache_position = kwargs.pop("cache_position")

            inputs_embeds = self.model._preprocess_prefill(
                input_ids=input_ids,
                pixel_values=pixel_values,
            )
            logits = self.model.language_model.prefill_decoder(
                inputs_embeds=inputs_embeds,
                cache_position=cache_position,
                block_tables=block_tables,
            ).logits
        else:
            padded_batch_size = kwargs.pop("padded_batch_size",
                                           self.decoder_batch_size)
            self.model.language_model.decoder = self.model.language_model.decoders[
                padded_batch_size]
            # FIXME position_ids, local_block_tables are required?
            logits = self.model.language_model.decoder(**kwargs).logits
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
                pixel_values = image_input["data"]
        else:
            pixel_values = None

        return pixel_values

    def _parse_and_validate_image_input(
            self, **kwargs: Any) -> Optional[PaliGemmaImageInputs]:
        pixel_values = kwargs.pop("pixel_values", None)
        image_embeds = kwargs.pop("image_embeds", None)
        config = self.vllm_config.model_config.hf_config
    
        if pixel_values is None and image_embeds is None:
            return None

        if pixel_values is not None:
            pixel_values = flatten_bn(pixel_values, concat=True)

            h = w = config.vision_config.image_size
            return PaliGemmaImagePixelInputs(type="pixel_values",
                                             data=pixel_values,
                                             resolve_bindings={
                                                 "h": h,
                                                 "w": w
                                             })

        if image_embeds is not None:
            image_embeds = flatten_bn(image_embeds, concat=True)

            return PaliGemmaImageEmbeddingInputs(
                type="image_embeds",
                data=image_embeds,
            )

        raise AssertionError("This line should be unreachable.")