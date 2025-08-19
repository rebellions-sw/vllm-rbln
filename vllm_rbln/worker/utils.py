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
"""A RBLN util class."""

import math
from typing import Optional

from vllm.config import ModelConfig, ParallelConfig


def get_maximum_num_blocks(
    model_config: ModelConfig,
    parallel_config: ParallelConfig,
    kvcache_block_size: int,
    nbits_per_param: Optional[int] = None,
    n_model_params: Optional[int] = None,
    kernel_size: Optional[int] = None,
    buffer: Optional[int] = None,
    num_runtimes: int = 2,
) -> int:
    # We are finding max_n_blocks(x) that satisfies the following equation:

    # available_dram - kernel_size - buffer
    #     - num_layers * 2 * tensor_parallel_size
    #     * align_2MB(
    #         x
    #         * block_size
    #         * align_64(head_dim)
    #         * math.ceil(num_key_value_heads / tensor_parallel_size)
    #         * 2
    #     ) > 0

    # This inequality can be rewritten as follows:

    # a - c * align_2MB(b * x) > 0
    # where
    #    a = available_dram - kernel_size - buffer
    #    b = block_size
    #         * align_64(head_dim)
    #         * math.ceil(num_key_value_heads / tensor_parallel_size) * 2
    #    c = num_layers * 2 * tensor_parallel_size

    # We can rewrite the inequality as follows:
    # k > align_2MB(b*x)
    # where
    #    k = a / c

    # After that, we can derive the following equation:
    # x = floor(2**21 / b * floor((k - 1) / 2**21))

    def align(x: int, nbytes: int) -> int:
        return int(math.ceil(x / nbytes) * nbytes)

    def align_2MB(x: int) -> int:
        return align(x, 2**21)

    num_layers = model_config.get_num_layers(parallel_config)
    head_dim = model_config.get_head_size()
    vocab_size = model_config.get_vocab_size()
    hidden_size = model_config.get_hidden_size()
    num_key_value_heads = model_config.get_num_kv_heads(parallel_config)
    tensor_parallel_size = parallel_config.tensor_parallel_size

    # TODO(jongho): Update if target npu is REBEL.
    ATOM_DRAM_NBYTES = 16 * 2**30
    ATOM_SYS_DRAM_NBYTES = 288 * 2**20
    available_dram = tensor_parallel_size * (ATOM_DRAM_NBYTES -
                                             ATOM_SYS_DRAM_NBYTES)

    if kernel_size is None:
        if n_model_params is None:
            raise ValueError("`n_model_params` should be specified \
                to estimate the kernel memory.")
        # Get estimated kernel size (approximated)
        lm_heads_params = align(vocab_size, 64) * hidden_size
        lm_heads_nbytes = (align_2MB(
            lm_heads_params * nbits_per_param // 8 / tensor_parallel_size) *
                           tensor_parallel_size)
        params = n_model_params - lm_heads_params
        layer_nbytes = (align_2MB(params * nbits_per_param // 8 / num_layers /
                                  tensor_parallel_size) * num_layers *
                        tensor_parallel_size)
        kernel_size = layer_nbytes + lm_heads_nbytes
    elif n_model_params is not None:
        raise ValueError(
            "Both `n_model_params` and `kernel_size` cannot be specified.")

    available_dram -= kernel_size

    if buffer is None:
        # TODO: Accurate buffer estimation
        buffer_per_runtime_per_core = 2**28  # 256MB per runtime
        # 1 for prefill, 1 for decoder
        buffer_per_core = buffer_per_runtime_per_core * num_runtimes
        buffer = buffer_per_core * tensor_parallel_size
    available_dram -= buffer

    b = kvcache_block_size * align(head_dim, 64) * math.ceil(
        num_key_value_heads / tensor_parallel_size) * 2
    c = num_layers * 2 * tensor_parallel_size
    k = available_dram / c
    max_n_blocks = math.floor(2**21 / b * math.floor((k - 1) / 2**21))

    return max_n_blocks
