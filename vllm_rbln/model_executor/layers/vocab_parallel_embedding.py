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

from typing import Optional

import torch
from vllm.distributed import (divide, get_tensor_model_parallel_rank,
                              get_tensor_model_parallel_world_size,
                              tensor_model_parallel_all_reduce)
from vllm.model_executor.layers.quantization.base_config import (
    QuantizationConfig, method_has_implemented_embedding)
from vllm.model_executor.layers.vocab_parallel_embedding import (
    DEFAULT_VOCAB_PADDING_SIZE, ParallelLMHead, UnquantizedEmbeddingMethod,
    VocabParallelEmbedding, get_masked_input_and_mask, pad_vocab_size)


def __vocab_parallel_embedding__init__(
    self,
    num_embeddings: int,
    embedding_dim: int,
    params_dtype: Optional[torch.dtype] = None,
    org_num_embeddings: Optional[int] = None,
    padding_size: int = DEFAULT_VOCAB_PADDING_SIZE,
    quant_config: Optional[QuantizationConfig] = None,
    prefix: str = "",
):
    torch.nn.Module.__init__(self)

    # Keep the input dimensions.
    if isinstance(self, ParallelLMHead):
        tp_rank = get_tensor_model_parallel_rank()
        self.tp_size = get_tensor_model_parallel_world_size()
    else:
        tp_rank = 0
        self.tp_size = 1
    self.num_embeddings = num_embeddings
    self.padding_size = padding_size
    self.org_vocab_size = org_num_embeddings or num_embeddings
    num_added_embeddings = num_embeddings - self.org_vocab_size
    self.org_vocab_size_padded = pad_vocab_size(self.org_vocab_size,
                                                self.padding_size)
    self.num_embeddings_padded = pad_vocab_size(
        self.org_vocab_size_padded + num_added_embeddings, self.padding_size)
    assert self.org_vocab_size_padded <= self.num_embeddings_padded

    self.shard_indices = self._get_indices(self.num_embeddings_padded,
                                           self.org_vocab_size_padded,
                                           self.num_embeddings,
                                           self.org_vocab_size, tp_rank,
                                           self.tp_size)
    self.embedding_dim = embedding_dim

    quant_method = None
    if quant_config is not None:
        quant_method = quant_config.get_quant_method(self, prefix=prefix)
    if quant_method is None:
        quant_method = UnquantizedEmbeddingMethod()

    # If we are making an embedding layer, then our quantization linear
    # method must implement the embedding operation. If we are another
    # layer type like ParallelLMHead, this is not important.
    is_embedding_layer = type(self) is VocabParallelEmbedding
    quant_method_implements_embedding = method_has_implemented_embedding(
        type(quant_method))
    if is_embedding_layer and not quant_method_implements_embedding:
        raise NotImplementedError(
            f"The class {type(quant_method).__name__} must implement "
            "the 'embedding' method, see UnquantizedEmbeddingMethod.")

    self.quant_method = quant_method

    if params_dtype is None:
        params_dtype = torch.get_default_dtype()
    # Divide the weight matrix along the vocaburaly dimension.
    self.num_added_embeddings = self.num_embeddings - self.org_vocab_size
    self.num_embeddings_per_partition = divide(self.num_embeddings_padded,
                                               self.tp_size)
    assert (self.shard_indices.num_elements_padded ==
            self.num_embeddings_per_partition)
    self.num_org_embeddings_per_partition = (
        self.shard_indices.org_vocab_end_index -
        self.shard_indices.org_vocab_start_index)
    self.num_added_embeddings_per_partition = (
        self.shard_indices.added_vocab_end_index -
        self.shard_indices.added_vocab_start_index)

    self.quant_method.create_weights(self,
                                     self.embedding_dim,
                                     [self.num_embeddings_per_partition],
                                     self.embedding_dim,
                                     self.num_embeddings_padded,
                                     params_dtype=params_dtype,
                                     weight_loader=self.weight_loader)


def __vocab_parallel_embedding_forward(self, input_):
    if self.tp_size > 1:
        # Build the mask.
        masked_input, input_mask = get_masked_input_and_mask(
            input_, self.shard_indices.org_vocab_start_index,
            self.shard_indices.org_vocab_end_index,
            self.shard_indices.num_org_vocab_padding,
            self.shard_indices.added_vocab_start_index,
            self.shard_indices.added_vocab_end_index)
    else:
        masked_input = input_
    # Get the embeddings.
    output_parallel = self.quant_method.embedding(self, masked_input.long())
    # Mask the output embedding.
    if self.tp_size > 1:
        output_parallel.masked_fill_(input_mask.unsqueeze(-1), 0)
        # Reduce across all the model parallel GPUs.
        output = tensor_model_parallel_all_reduce(output_parallel)
    else:
        output = output_parallel
    return output


def __parallel_lm_head_tie_weights(self, embed_tokens: VocabParallelEmbedding):
    """Tie the weights with word embeddings."""
    # GGUF quantized embed_tokens.
    if self.quant_config and self.quant_config.get_name() == "gguf":
        return embed_tokens
    else:
        if self.tp_size < 2:
            self.weight = embed_tokens.weight
        return self


VocabParallelEmbedding.__init__ = __vocab_parallel_embedding__init__
VocabParallelEmbedding.forward = __vocab_parallel_embedding_forward
ParallelLMHead.tie_weights = __parallel_lm_head_tie_weights
