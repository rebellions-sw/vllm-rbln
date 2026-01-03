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


def rbln_guard_filter_fn(guard_entries):
    """
    Keep tensor guards (unsafe) + additionally keep the python guard
    related to num_logprobs.
    """
    # 1) Start from PyTorch's built-in tensor-guard keep policy
    base_keep = torch.compiler.keep_tensor_guards_unsafe(guard_entries)

    # 2) Add our extra keep condition
    out = []
    for keep, e in zip(base_keep, guard_entries):
        name = getattr(e, "name", "") or ""
        keep_max_num_logprobs = ("num_logprobs" in name)
        out.append(bool(keep or keep_max_num_logprobs))
    return out
