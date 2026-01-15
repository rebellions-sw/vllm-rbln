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

from .bucketing_manager import RBLNBucketingManager
from .exponential_bucketing_manager import ExponentialBucketingManager
from .linear_bucketing_manager import LinearBucketingManager

__all__ = [
    "RBLNBucketingManager",
    "ExponentialBucketingManager",
    "LinearBucketingManager",
]


def get_bucketing_manager_class(strategy: str) -> type[RBLNBucketingManager]:
    if strategy == "exponential" or strategy == "exp":
        return ExponentialBucketingManager
    elif strategy == "linear":
        return LinearBucketingManager
    else:
        raise ValueError(
            f"Invalid bucketing strategy: {strategy}. "
            "Valid strategies are [exponential, exp, linear].", )
