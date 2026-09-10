# Copyright 2026 Rebellions Inc. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at:
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from dataclasses import dataclass
from typing import Any

import torch
from vllm.sequence import IntermediateTensors


@dataclass(frozen=True)
class InputLayout:
    num_reqs: int
    num_reqs_padded: int
    query_len: int
    query_len_padded: int
    input_pad_value: int = 0
    position_pad_value: int = 0
    hidden_state_pad_value: float = 0.0
    token_index_pad_value: int = 0
    # Rows of the gather indices; token_indices defaults to one per request.
    num_token_indices: int | None = None
    num_bonus_token_indices: int | None = None

    @property
    def shape(self) -> tuple[int, int]:
        return (self.num_reqs_padded, self.query_len_padded)

    @property
    def token_indices_size(self) -> int:
        if self.num_token_indices is None:
            return self.num_reqs_padded
        return self.num_token_indices


@dataclass
class InputBuffer:
    input_ids: torch.Tensor
    positions: torch.Tensor


@dataclass(slots=True)
class StagedModelInputs:
    input_ids: torch.Tensor
    positions: torch.Tensor
    intermediate_tensors: IntermediateTensors | None
    inputs_embeds: torch.Tensor | None
    token_indices: torch.Tensor | None
    # For Eagle3 drafter
    hidden_states: torch.Tensor | None = None
    bonus_token_indices: torch.Tensor | None = None

    def as_kwargs(self) -> dict[str, Any]:
        return {
            "input_ids": self.input_ids,
            "positions": self.positions,
            "intermediate_tensors": self.intermediate_tensors,
            "inputs_embeds": self.inputs_embeds,
            "token_indices": self.token_indices,
            "bonus_token_indices": self.bonus_token_indices,
        }


class InputStager:
    def __init__(self, device: torch.device):
        self.device = device
        self._buffers: dict[tuple, InputBuffer] = {}
        self._hidden_state_buffers: dict[tuple, torch.Tensor] = {}
        self._token_indices_buffers: dict[tuple[torch.dtype, int], torch.Tensor] = {}
        self._bonus_token_indices_buffers: dict[
            tuple[torch.dtype, int], torch.Tensor
        ] = {}

    def stage(
        self,
        *,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        token_indices: torch.Tensor | None = None,
        bonus_token_indices: torch.Tensor | None = None,
        hidden_states: torch.Tensor | None = None,
        layout: InputLayout,
    ) -> StagedModelInputs:
        buf = self._get_or_create_buffer(layout, input_ids, positions)

        assert buf.input_ids.shape == layout.shape
        buf.input_ids.fill_(layout.input_pad_value)
        buf.input_ids[: layout.num_reqs, : layout.query_len].copy_(
            input_ids,
            non_blocking=True,
        )

        buf.positions.fill_(layout.position_pad_value)
        buf.positions[: layout.num_reqs, : layout.query_len].copy_(
            positions,
            non_blocking=True,
        )

        return StagedModelInputs(
            input_ids=buf.input_ids,
            positions=buf.positions,
            intermediate_tensors=intermediate_tensors,
            inputs_embeds=inputs_embeds,
            token_indices=self._stage_token_indices(token_indices, layout),
            hidden_states=self._stage_hidden_states(hidden_states, layout),
            bonus_token_indices=self._stage_index(
                bonus_token_indices,
                layout.num_bonus_token_indices,
                layout.token_index_pad_value,
                self._bonus_token_indices_buffers,
            ),
        )

    def _get_or_create_buffer(
        self,
        layout: InputLayout,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
    ) -> InputBuffer:
        key = (
            layout.shape,
            input_ids.dtype,
            positions.dtype,
        )
        if key in self._buffers:
            return self._buffers[key]

        buf = InputBuffer(
            input_ids=torch.empty(
                layout.shape,
                dtype=input_ids.dtype,
                device=self.device,
            ),
            positions=torch.empty(
                layout.shape,
                dtype=positions.dtype,
                device=self.device,
            ),
        )
        self._buffers[key] = buf
        return buf

    def _stage_hidden_states(
        self,
        hidden_states: torch.Tensor | None,
        layout: InputLayout,
    ) -> torch.Tensor | None:
        if hidden_states is None:
            return None

        key = (layout.shape, hidden_states.dtype, hidden_states.shape[-1])
        if (buf := self._hidden_state_buffers.get(key)) is None:
            buf = torch.empty(
                (*layout.shape, hidden_states.shape[-1]),
                dtype=hidden_states.dtype,
                device=self.device,
            )
            self._hidden_state_buffers[key] = buf

        buf[layout.num_reqs :].fill_(layout.hidden_state_pad_value)
        buf[: layout.num_reqs, layout.query_len :].fill_(layout.hidden_state_pad_value)
        buf[: layout.num_reqs, : layout.query_len].copy_(
            hidden_states,
            non_blocking=True,
        )
        return buf

    def _stage_token_indices(
        self,
        token_indices: torch.Tensor | None,
        layout: InputLayout,
    ) -> torch.Tensor | None:
        if token_indices is None:
            return None
        return self._stage_index(
            token_indices,
            layout.token_indices_size,
            layout.token_index_pad_value,
            self._token_indices_buffers,
        )

    def _stage_index(
        self,
        indices: torch.Tensor | None,
        size: int | None,
        pad_value: int,
        buffers: dict[tuple[torch.dtype, int], torch.Tensor],
    ) -> torch.Tensor | None:
        """Copy a 1-D index into a fixed-size device buffer (keyed by dtype and
        size so the graph input address stays put), padding the tail."""
        if indices is None:
            return None
        assert size is not None
        num_indices = indices.shape[0]
        assert num_indices <= size

        key = (indices.dtype, size)
        if (buf := buffers.get(key)) is None:
            buf = torch.empty(size, dtype=indices.dtype, device=self.device)
            buffers[key] = buf

        buf.fill_(pad_value)
        buf[:num_indices].copy_(indices, non_blocking=True)
        return buf
