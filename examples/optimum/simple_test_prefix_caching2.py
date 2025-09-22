import typing

import fire
import torch

from optimum.rbln import RBLNLlamaForCausalLM


def main(
    model_id: str = "meta-llama/Llama-3.2-3B-Instruct",
    batch_size: int = 2,
    from_transformers: bool = False,
    tensor_parallel_size: typing.Optional[int] = 1,
):
    if from_transformers:
        model = RBLNLlamaForCausalLM.from_pretrained(
            model_id=model_id,
            export=True,
            # The following arguments are specific to RBLN compilation
            rbln_batch_size=batch_size,
            rbln_max_seq_len=4096,
            rbln_tensor_parallel_size=tensor_parallel_size,
            torch_dtype="auto",
            num_hidden_layers=1,
        )
        model.save_pretrained("test_llama_model")
    else:
        model = RBLNLlamaForCausalLM.from_pretrained("test_llama_model", export=False)

    prefill_runtime = model.prefill_decoder.runtime
    CACHED_LEN = 128
    TOTAL_LEN = 256

    # 0 ~ 127 까지 0번 블록 inference
    input_ids = torch.arange(0, CACHED_LEN, dtype=torch.int64).unsqueeze(0)
    cache_position = torch.arange(0, CACHED_LEN, dtype=torch.int32).unsqueeze(0)
    block_tables = torch.tensor([0], dtype=torch.int16)
    query_position = torch.tensor(CACHED_LEN - 1, dtype=torch.int16)
    out1 = prefill_runtime(input_ids, cache_position, block_tables, query_position=query_position)

    # 128 ~ 255 까지 0번 블록 inference
    input_ids = torch.arange(CACHED_LEN, TOTAL_LEN, dtype=torch.int64).unsqueeze(0)
    cache_position = torch.arange(CACHED_LEN, TOTAL_LEN, dtype=torch.int32).unsqueeze(0)
    out2 = prefill_runtime(input_ids, cache_position, block_tables, query_position=query_position)

    # 0번 block => 1번 block에 복사.
    prefill_runtime._copy_kv_cache(0, 1, CACHED_LEN)  # src, dst, cached_lengths

    # 1번 블록도 128 ~ 255 까지 inference
    block_tables = torch.tensor([1], dtype=torch.int16)
    out3 = prefill_runtime(input_ids, cache_position, block_tables, query_position=query_position)

    # 우리가 기대하는 것은 out3 & out2 가 같아야함.
    print(torch.allclose(out2, out3))
    print(out2)
    print(out3)


if __name__ == "__main__":
    fire.Fire(main)