from vllm.v1.kv_cache_interface import AttentionSpec
from vllm.utils import cdiv

@dataclass(frozen=True)
class RBLNSlidingWindowImageSpec(AttentionSpec):
    sliding_window: int

    def __post_init__(self):
        assert not self.use_mla, "MLA is not supported for sliding window"

    def max_memory_usage_bytes(self, vllm_config: VllmConfig) -> int:
        max_model_len = vllm_config.model_config.max_model_len
        # max_num_batched_tokens = (
        #     vllm_config.scheduler_config.max_num_batched_tokens)

        # # During chunked prefill, we allocate KV cache for the last
        # # `self.sliding_window-1` computed tokens plus the newly scheduled
        # # tokens. And we won't allocate KV cache for more than `max_model_len`
        # # tokens.
        # num_tokens = min(self.sliding_window - 1 + max_num_batched_tokens,
        #                  max_model_len)

        # +1 here because the sliding window may not start from the beginning
        # of the block. For example, if the block size is 4 and num_token
        # is 4, we need two blocks [XXCD] [EF] to store the sliding
        # window [CDEF] of 6 tokens.
        return (cdiv(max_model_len, self.block_size) + 1) * self.page_size_bytes


class RBLNSlidingWindowSpec(AttentionSpec):
    sliding_window: int

    def __post_init__(self):
        assert not self.use_mla, "MLA is not supported for sliding window"

    def max_memory_usage_bytes(self, vllm_config: VllmConfig) -> int:
        max_model_len = vllm_config.model_config.max_model_len

        # NOTE For now, eager attn is only available
        return cdiv(self.sliding_window, self.block_size) * self.page_size_bytes
