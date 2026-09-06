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
from abc import ABC, abstractmethod
from dataclasses import dataclass, replace
from typing import Any

import torch
from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.model_executor.models.interfaces import MultiModalEmbeddings
from vllm.model_executor.models.qwen2_5_vl import (
    Qwen2_5_VLImageEmbeddingInputs,
    Qwen2_5_VLImagePixelInputs,
    Qwen2_5_VLVideoEmbeddingInputs,
    Qwen2_5_VLVideoPixelInputs,
)
from vllm.model_executor.models.qwen2_vl import (
    Qwen2VLImageEmbeddingInputs,
    Qwen2VLImagePixelInputs,
    Qwen2VLVideoEmbeddingInputs,
    Qwen2VLVideoPixelInputs,
)

from .base import ModelInputForRBLN
from .model_base import (
    RBLNOptimumDecoderMixin,
    RBLNOptimumModelBase,
    RBLNOptimumMultimodalMixin,
)

logger = init_logger(__name__)


@dataclass(frozen=True)
class ModalitySpec:
    """Per-modality kwarg keys and the config attribute for its placeholder id."""

    name: str  # "image" | "video"
    grid_key: str  # grid_thw kwarg key
    pixel_key: str  # pixel-values kwarg key
    token_attr: str  # config attribute holding the placeholder token id


MODALITIES: tuple[ModalitySpec, ModalitySpec] = (
    ModalitySpec("image", "image_grid_thw", "pixel_values", "image_token_id"),
    ModalitySpec("video", "video_grid_thw", "pixel_values_videos", "video_token_id"),
)


def split_by_grid_thw(
    embeds: torch.Tensor, grid_thw: torch.Tensor
) -> list[torch.Tensor]:
    """Cut the vision encoder's concatenated output back into per-item tensors.

    Each item covers ``t * h * w`` patches merged ``k x k`` into one token; the
    merge unit ``k**2`` is read off the output itself so this also works on an
    EC producer, which loads only the encoder and has no text config.
    """
    patches = grid_thw.prod(dim=-1)
    merge_unit = int(patches.sum()) // embeds.shape[0]
    return list(embeds.split((patches // merge_unit).tolist()))


class RBLNOptimumQwenVLForConditionalGeneration(
    RBLNOptimumModelBase,
    RBLNOptimumMultimodalMixin,
    RBLNOptimumDecoderMixin,
    ABC,
):
    """
    Unified class for both Qwen2-VL and Qwen2.5-VL models.
    Automatically detects model type based on the model configuration.
    """

    @classmethod
    def get_placeholder_str(cls, modality: str, i: int) -> str | None:
        if modality.startswith("image"):
            return "<|vision_start|><|image_pad|><|vision_end|>"
        if modality.startswith("video"):
            return "<|vision_start|><|video_pad|><|vision_end|>"

        raise ValueError("Only image or video modality is supported")

    def __init__(
        self,
        vllm_config: VllmConfig,
    ) -> None:
        super().__init__(vllm_config=vllm_config)
        if self._is_ec_producer_only():
            return
        assert self.kv_block_adapter is not None
        self.setup_decoder_mixin(
            attn_impl=self.attn_impl,
            vocab_size=self.model_config.get_vocab_size,
            use_multiple_decoder=getattr(
                self.model.rbln_config, "use_multiple_decoder", False
            ),
            default_batch_size=self.scheduler_config.max_num_seqs,
            decoder_batch_sizes=self.model.rbln_config.decoder_batch_sizes,
        )

    def get_prefill_decoder(self):
        return self.model.prefill_decoder

    def get_language_model(self):
        return self.model

    @abstractmethod
    def _add_model_specific_args(self, preprocess_args: dict, video_input: Any):
        """
        Add model-specific arguments to preprocessing args.

        Args:
            preprocess_args: Dictionary of preprocessing arguments to modify
            video_input: Video input data
        """
        pass

    @abstractmethod
    def _create_image_pixel_inputs(
        self, pixel_values: torch.Tensor, image_grid_thw: torch.Tensor
    ) -> Any:
        """Create image pixel inputs based on model type"""
        pass

    @abstractmethod
    def _create_image_embedding_inputs(
        self, image_embeds: torch.Tensor, image_grid_thw: torch.Tensor
    ) -> Any:
        """Create image embedding inputs based on model type"""
        pass

    @abstractmethod
    def _create_video_pixel_inputs(
        self,
        pixel_values_videos: torch.Tensor,
        video_grid_thw: torch.Tensor,
        second_per_grid_ts: torch.Tensor | None,
    ) -> Any:
        """Create video pixel inputs based on model type"""
        pass

    @abstractmethod
    def _create_video_embedding_inputs(self, video_embeds, video_grid_thw) -> Any:
        """Create video embedding inputs based on model type"""
        pass

    def _parse_and_validate_image_input(self, **kwargs: Any) -> Any | None:
        pixel_values = kwargs.pop("pixel_values", None)
        image_embeds = kwargs.pop("image_embeds", None)
        image_grid_thw = kwargs.pop("image_grid_thw", None)

        if pixel_values is None and image_embeds is None:
            return None

        if pixel_values is not None:
            return self._create_image_pixel_inputs(
                pixel_values=pixel_values, image_grid_thw=image_grid_thw
            )

        if image_embeds is not None:
            return self._create_image_embedding_inputs(
                image_embeds=image_embeds, image_grid_thw=image_grid_thw
            )

        # fallback return if both are None
        return None

    def _parse_and_validate_video_input(self, **kwargs: object) -> Any | None:
        pixel_values_videos = kwargs.pop("pixel_values_videos", None)
        video_embeds = kwargs.pop("video_embeds", None)
        video_grid_thw = kwargs.pop("video_grid_thw", None)
        second_per_grid_ts = kwargs.pop("second_per_grid_ts", None)

        if pixel_values_videos is None and video_embeds is None:
            return None

        if pixel_values_videos is not None:
            return self._create_video_pixel_inputs(
                pixel_values_videos, video_grid_thw, second_per_grid_ts
            )

        if video_embeds is not None:
            return self._create_video_embedding_inputs(video_embeds, video_grid_thw)

        # fallback return if both are None
        return None

    def _process_image_input(self, image_input) -> list[torch.Tensor]:
        if image_input is None or image_input.get("type") != "pixel_values":
            return []
        grid_thw = image_input["image_grid_thw"]
        embeds = self.model.visual(image_input["pixel_values"], grid_thw=grid_thw)
        return split_by_grid_thw(embeds, grid_thw)

    def _process_video_input(self, video_input) -> list[torch.Tensor]:
        if video_input is None or video_input.get("type") != "pixel_values_videos":
            return []
        grid_thw = video_input["video_grid_thw"]
        embeds = self.model.visual(
            video_input["pixel_values_videos"], grid_thw=grid_thw
        )
        return split_by_grid_thw(embeds, grid_thw)

    def embed_multimodal(self, **kwargs: object) -> MultiModalEmbeddings:
        """One 2D tensor per image or video item, in kwargs order. The runner
        batches one modality per call and caches each item by mm_hash."""
        image_input = self._parse_and_validate_image_input(**kwargs)
        video_input = self._parse_and_validate_video_input(**kwargs)
        return [
            *self._process_image_input(image_input),
            *self._process_video_input(video_input),
        ]

    def _image_token_id(self) -> int:
        return self.model.config.image_token_id

    def _embed_text_tokens(
        self, input_ids: torch.Tensor, is_multimodal: torch.Tensor
    ) -> torch.Tensor:
        return self.model.embed_tokens(input_ids)

    def build_prefill_forward_inputs(
        self,
        model_input: ModelInputForRBLN,
        mrope_position_deltas: dict[str, float],
    ) -> ModelInputForRBLN:
        """Scatter the multimodal embeddings, then add the MRoPE positions and
        record the request's rope delta for its decode steps."""
        model_input = super().build_prefill_forward_inputs(
            model_input, mrope_position_deltas
        )
        position_embed, rope_deltas = self._build_prefill_position_embed(model_input)
        mrope_position_deltas[model_input.running_requests_ids[0]] = rope_deltas.item()
        return replace(model_input, position_embed=position_embed)

    def _build_prefill_position_embed(
        self, model_input: ModelInputForRBLN
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """MRoPE ``(position_embed, rope_deltas)`` for prefill, unified across
        full and partial prefix-cache hits.

        Each multimodal item shifts every later token's position, so MRoPE
        positions depend on the whole prompt layout and cannot be computed from
        the uncached tail alone. They are therefore always computed over the
        full prompt with the encoder skipped (grids only), then sliced to the
        uncached window ``[num_cached:]``:

        - full prefill: ``num_cached == 0``, so the whole prompt is kept;
        - partial hit: only the uncached tail is kept.

        ``rope_deltas`` is over the full sequence (used for decode positions).
        """
        partial = model_input.partial_prefix
        if partial is not None:
            full_input_ids = partial.full_input_tokens
            num_cached = partial.num_cached_tokens
            mm_kwargs = partial.mrope_mm_kwargs
        else:
            full_input_ids = model_input.input_tokens
            num_cached = 0
            mm_kwargs = model_input.multi_modal_kwargs

        image_input = None
        video_input = None
        if mm_kwargs:
            image_input = self._parse_and_validate_image_input(**mm_kwargs)
            video_input = self._parse_and_validate_video_input(**mm_kwargs)

        attention_mask = torch.ones_like(full_input_ids)
        params = self._compute_mrope_position(
            full_input_ids, attention_mask, image_input, video_input
        )
        # position_embed: [2, batch, 1, N, head_dim]; slice the sequence (dim=-2)
        # to the uncached window (whole prompt when num_cached == 0).
        position_embed = params["position_embed"][..., num_cached:, :]
        return position_embed, params["rope_deltas"]

    def _compute_mrope_position(
        self, input_ids, attention_mask, image_input, video_input
    ) -> dict:
        """MRoPE positions only: run ``get_rope_index`` with grids but no
        ``pixel_values`` (encoder skipped). Returns ``{position_embed,
        rope_deltas}``.
        """
        preprocess_args = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }
        for spec, mm_input in zip(MODALITIES, (image_input, video_input)):
            preprocess_args[spec.pixel_key] = None
            preprocess_args[spec.grid_key] = (
                mm_input[spec.grid_key] if mm_input is not None else None
            )
        # second_per_grid_ts (video, Qwen2.5-VL) feeds get_rope_index too.
        self._add_model_specific_args(preprocess_args, video_input)

        outputs = self.model._preprocess_prefill(**preprocess_args)
        # outputs[1]/[2] = position_embed/rope_deltas across all variants; the
        # rest of the tuple's arity differs, so don't unpack it.
        return {"position_embed": outputs[1], "rope_deltas": outputs[2]}

    def compute_decode_position_embed(
        self,
        model_input: ModelInputForRBLN,
        mrope_position_deltas: dict[str, float],
    ) -> torch.Tensor:
        """Decode-step MRoPE: advance each request's position from its stored
        delta (``cache_position + mrope_position_delta``) and return the position
        embeddings (cos/sin) laid out like the decode batch: each request at its
        row, zeros in the padding rows. Mirrors upstream vLLM's
        ``get_next_input_positions_tensor``.
        """
        cache_position = model_input.input_positions
        running_requests_ids = model_input.running_requests_ids
        rows: torch.Tensor | slice = (
            slice(0, len(running_requests_ids))
            if model_input.batch_rows is None
            else model_input.batch_rows
        )
        row_ids = (
            range(len(running_requests_ids))
            if model_input.batch_rows is None
            else model_input.batch_rows.tolist()
        )

        position_embeds = []
        for row, request_id in zip(row_ids, running_requests_ids):
            delta = cache_position[row] + mrope_position_deltas[request_id]
            position_ids = torch.arange(1).view(1, -1)
            position_ids = position_ids.add(delta)
            position_ids = position_ids.unsqueeze(0).expand(3, -1, -1)
            position_embed = self.model._get_position_embeddings(
                torch.zeros(1, dtype=self.dtype), position_ids
            )
            position_embeds.append(position_embed)
        embeds = torch.cat(position_embeds, dim=1)

        shape = list(embeds.shape)
        shape[1] = model_input.padded_batch_size
        out = embeds.new_zeros(shape)
        out[:, rows] = embeds
        return out

    def forward(self, model_input: ModelInputForRBLN, **kwargs) -> torch.Tensor:
        if model_input.is_prompt:
            return self.model.prefill_decoder(
                inputs_embeds=model_input.inputs_embeds,
                position_embed=model_input.position_embed,
                block_tables=model_input.block_tables,
                cache_position=model_input.input_positions,
            ).logits

        self.model.decoder = self.model.decoders[model_input.padded_batch_size]
        logits = self.model.decoder(
            inputs_embeds=self.model.embed_tokens(model_input.input_tokens),
            cache_position=model_input.input_positions,
            position_embed=model_input.position_embed,
            block_tables=model_input.block_tables,
        ).logits
        return logits[: len(model_input.running_requests_ids)]


class RBLNOptimumQwen2_5_VLForConditionalGeneration(
    RBLNOptimumQwenVLForConditionalGeneration
):
    def _add_model_specific_args(self, preprocess_args: dict, video_input: Any):
        """Add second_per_grid_ts for Qwen2.5-VL"""
        if video_input is not None:
            second_per_grid_ts = video_input.get("second_per_grid_ts", None)
            if second_per_grid_ts is not None:
                preprocess_args["second_per_grid_ts"] = second_per_grid_ts

    def _create_image_pixel_inputs(self, pixel_values, image_grid_thw):
        return Qwen2_5_VLImagePixelInputs(
            type="pixel_values",
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
        )

    def _create_image_embedding_inputs(self, image_embeds, image_grid_thw):
        return Qwen2_5_VLImageEmbeddingInputs(
            type="image_embeds",
            image_embeds=image_embeds,
            image_grid_thw=image_grid_thw,
        )

    def _create_video_pixel_inputs(
        self,
        pixel_values_videos: torch.Tensor,
        video_grid_thw: torch.Tensor,
        second_per_grid_ts=torch.Tensor | None,
    ):
        if second_per_grid_ts is None:
            raise ValueError(
                "second_per_grid_ts is required for Qwen2.5-VL video inputs."
            )
        return Qwen2_5_VLVideoPixelInputs(
            type="pixel_values_videos",
            pixel_values_videos=pixel_values_videos,
            video_grid_thw=video_grid_thw,
            second_per_grid_ts=second_per_grid_ts,
        )

    def _create_video_embedding_inputs(self, video_embeds, video_grid_thw):
        return Qwen2_5_VLVideoEmbeddingInputs(
            type="video_embeds",
            video_embeds=video_embeds,
            video_grid_thw=video_grid_thw,
        )


class RBLNOptimumQwen2VLForConditionalGeneration(
    RBLNOptimumQwenVLForConditionalGeneration
):
    def _add_model_specific_args(self, preprocess_args: dict, video_input: Any):
        """Qwen2-VL doesn't need additional arguments"""
        pass

    def _create_image_pixel_inputs(self, pixel_values, image_grid_thw):
        return Qwen2VLImagePixelInputs(
            type="pixel_values",
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
        )

    def _create_image_embedding_inputs(self, image_embeds, image_grid_thw):
        return Qwen2VLImageEmbeddingInputs(
            type="image_embeds",
            image_embeds=image_embeds,
            image_grid_thw=image_grid_thw,
        )

    def _create_video_pixel_inputs(
        self,
        pixel_values_videos: torch.Tensor,
        video_grid_thw: torch.Tensor,
        second_per_grid_ts: torch.Tensor | None,
    ):
        # NOTE Qwen2-VL doesn't use second_per_grid_ts
        return Qwen2VLVideoPixelInputs(
            type="pixel_values_videos",
            pixel_values_videos=pixel_values_videos,
            video_grid_thw=video_grid_thw,
        )

    def _create_video_embedding_inputs(self, video_embeds, video_grid_thw):
        return Qwen2VLVideoEmbeddingInputs(
            type="video_embeds",
            video_embeds=video_embeds,
            video_grid_thw=video_grid_thw,
        )
