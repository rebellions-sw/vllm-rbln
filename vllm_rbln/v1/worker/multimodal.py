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
from collections import defaultdict
from typing import TYPE_CHECKING, cast

import numpy as np
from vllm.multimodal.inputs import (BatchedTensorInputs, MultiModalKwargs,
                                    NestedTensors)
from vllm.utils import LazyLoader, is_list_of

if TYPE_CHECKING:
    import torch
    import torch.types

else:
    torch = LazyLoader("torch", globals(), "torch")


class RBLNOptimumMultiModalKwargs(MultiModalKwargs):

    @staticmethod
    def _try_stack(nested_tensors: NestedTensors,
                   pin_memory: bool = False) -> NestedTensors:
        """
        Stack the inner dimensions that have the same shape in
        a nested list of tensors.

        Thus, a dimension represented by a list means that the inner
        dimensions are different for each element along that dimension.
        """
        if isinstance(nested_tensors, torch.Tensor):
            return nested_tensors

        # TODO: Remove these once all models have been migrated
        if isinstance(nested_tensors, np.ndarray):
            return torch.from_numpy(nested_tensors)
        if isinstance(nested_tensors, (int, float)):
            return torch.tensor(nested_tensors)

        stacked = [
            RBLNOptimumMultiModalKwargs._try_stack(t, pin_memory)
            for t in nested_tensors
        ]
        if not is_list_of(stacked, torch.Tensor, check="all"):
            # Only tensors (not lists) can be stacked.
            return stacked

        tensors_ = cast(list[torch.Tensor], stacked)
        # if len(tensors_) == 1:
        #     # An optimization when `tensors_` contains only one tensor:
        #     # - produce exactly same result as `torch.stack(tensors_)`
        #     # - will achieve zero-copy if the tensor is contiguous
        #     return tensors_[0].unsqueeze(0).contiguous()
        # NOTE(eunji): This optimization in vllm-rbln is not working.
        # So I turned it off.

        if any(t.shape != tensors_[0].shape for t in tensors_):
            # The tensors have incompatible shapes and can't be stacked.
            return tensors_

        outputs = torch.empty(len(tensors_),
                              *tensors_[0].shape,
                              dtype=tensors_[0].dtype,
                              device=tensors_[0].device,
                              pin_memory=pin_memory)
        return torch.stack(tensors_, out=outputs)

    @staticmethod
    def batch(inputs_list: list["RBLNOptimumMultiModalKwargs"],
              pin_memory: bool = False) -> BatchedTensorInputs:
        """
        Batch multiple inputs together into a dictionary.

        The resulting dictionary has the same keys as the inputs.
        If the corresponding value from each input is a tensor and they all
        share the same shape, the output value is a single batched tensor;
        otherwise, the output value is a list containing the original value
        from each input.
        """
        if len(inputs_list) == 0:
            return {}

        # We need to consider the case where each item in the batch
        # contains different modalities (i.e. different keys).
        item_lists = defaultdict[str, list[NestedTensors]](list)

        for inputs in inputs_list:
            for k, v in inputs.items():
                item_lists[k].append(v)
        return {
            k: RBLNOptimumMultiModalKwargs._try_stack(item_list, pin_memory)
            for k, item_list in item_lists.items()
        }
