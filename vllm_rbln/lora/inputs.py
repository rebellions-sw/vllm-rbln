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


class LoRAInputs:
    sampler_indices_padded: torch.Tensor

    @staticmethod
    def set_sampler_indices_padded(
            sampler_indices_padded: torch.Tensor) -> None:
        LoRAInputs.sampler_indices_padded = sampler_indices_padded

    @staticmethod
    def get_sampler_indices_padded() -> torch.Tensor:
        return LoRAInputs.sampler_indices_padded
