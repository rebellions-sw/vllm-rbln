import torch
from typing import Optional

from vllm.distributed import (tensor_model_parallel_all_gather,
                              tensor_model_parallel_gather)
from vllm.model_executor.layers.logits_processor import (
    LogitsProcessor
    )
from vllm.model_executor.layers.vocab_parallel_embedding import (
    VocabParallelEmbedding)


def logits_processor_get_logits(
    self,
    hidden_states: torch.Tensor,
    lm_head: VocabParallelEmbedding,
    embedding_bias: Optional[torch.Tensor],
) -> Optional[torch.Tensor]:
    # Get the logits for the next tokens.
    logits = lm_head.quant_method.apply(lm_head,
                                        hidden_states,
                                        bias=embedding_bias)
    return logits

def logits_processor_gather_logits(self, logits: torch.Tensor) -> torch.Tensor:
    """gather/all-gather the logits tensor across model parallel group."""
    if self.use_all_gather:
        logits = tensor_model_parallel_all_gather(logits)
    else:
        # None may be returned for rank > 0
        logits = tensor_model_parallel_gather(logits)

    # Remove paddings in vocab (if any).
    if logits is not None:
        logits = logits[..., :self.org_vocab_size]
    return logits

LogitsProcessor._get_logits = logits_processor_get_logits
LogitsProcessor._gather_logits = logits_processor_gather_logits
