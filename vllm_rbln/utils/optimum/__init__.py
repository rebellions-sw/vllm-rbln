from optimum.rbln import RBLNModelConfig
from vllm.config import VllmConfig


def check_rbln_config(rbln_config: RBLNModelConfig, vllm_config: VllmConfig):
    submodules = rbln_config.submodules
    batch_size = rbln_config.batch_size
    kvcache_partition_len = getattr(rbln_config, "kvcache_partition_len", None)
    max_seq_len = getattr(rbln_config, "max_seq_len", None)
    dec_max_seq_len = getattr(rbln_config, "dec_max_seq_len", None)

    max_model_len = vllm_config.model_config.max_model_len
    max_num_seqs = vllm_config.scheduler_config.max_num_seqs
    block_size = vllm_config.cache_config.block_size
    max_num_batched_tokens = vllm_config.scheduler_config.max_num_batched_tokens

    # NOTE It is based on the decoder submodule is only one.
    for submodule in submodules:
        submodule_config = getattr(rbln_config, submodule)
        if kvcache_partition_len is None:
            kvcache_partition_len = getattr(submodule_config,
                                            "kvcache_partition_len", None)
        if max_seq_len is None:
            max_seq_len = getattr(submodule_config, "max_seq_len", None)

        if dec_max_seq_len is None:
            dec_max_seq_len = getattr(submodule_config, "dec_max_seq_len",
                                      None)

        submodule_batch_size = getattr(submodule_config, "batch_size", None)
        # FIXME
        if submodule_batch_size is not None \
            and batch_size != submodule_batch_size:
            batch_size = submodule_batch_size

    if kvcache_partition_len is None:
        if max_seq_len is not None:
            assert block_size == max_seq_len, (
                f"`block_size({block_size})` must match "
                f"`max_seq_len({max_seq_len})` "
                "of the compiled RBLN model.")
        elif dec_max_seq_len is not None:
            assert block_size == dec_max_seq_len, (
                f"`block_size({block_size})` must match "
                f"`dec_max_seq_len({dec_max_seq_len})` "
                "of the compiled RBLN model.")
    else:
        assert block_size == kvcache_partition_len, (
            f"`block_size({block_size})` must match "
            f"the `kvcache_partition_len({kvcache_partition_len})` "
            "of the compiled RBLN model.")

    if dec_max_seq_len is not None:  # encoder-decoder
        assert max_num_batched_tokens == dec_max_seq_len, (
            f"`max_num_batched_tokens({max_num_batched_tokens})` "
            f"must match the `dec_max_seq_len({dec_max_seq_len})` "
            "of the compiled RBLM model.")
        assert max_model_len == dec_max_seq_len, (
            f"`max_model_len({max_model_len})` must match "
            f"the `dec_max_seq_len({dec_max_seq_len})` "
            "of the compiled RBLM model.")
    elif max_seq_len is not None:  # including decoder
        assert max_num_batched_tokens == max_seq_len, (
            f"`max_num_batched_tokens({max_num_batched_tokens})` "
            f"must match the `max_seq_len({max_seq_len})` "
            "of the compiled RBLM model.")
        assert max_model_len == max_seq_len, (
            f"`max_model_len({max_model_len})` must match "
            f"the `max_seq_len({max_seq_len})` "
            "of the compiled RBLM model.")

    assert max_num_seqs == batch_size, (
        f"`max_num_seqs({max_num_seqs})` must match "
        f"the `batch_size({batch_size})` of the compiled RBLM model.")
