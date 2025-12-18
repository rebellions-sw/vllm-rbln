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

from typing import Union

import torch


def pad(x: torch.Tensor,
        dim: int,
        target_len: int,
        pad_value: Union[int, float] = 0) -> torch.Tensor:
    """Pad along the given dimension to target_len using pad_value."""
    current = x.size(dim)
    if current >= target_len:
        # NOTE: dynamo distinguishes views and non-views for inputs,
        # so ensure that the output is always a non-view.
        return x if x._base is None else x.clone()

    pad_shape = list(x.shape)
    pad_shape[dim] = target_len - current
    pad = torch.full(pad_shape, pad_value, dtype=x.dtype, device=x.device)
    return torch.cat([x, pad], dim=dim)
