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


class LoRAMask:
    lora_mask: torch.Tensor  # [batch_size, max_loras * max_rank]

    @staticmethod
    def set_lora_mask(mask: torch.Tensor) -> None:
        LoRAMask.lora_mask = mask

    @staticmethod
    def get_lora_mask() -> torch.Tensor:
        return LoRAMask.lora_mask
