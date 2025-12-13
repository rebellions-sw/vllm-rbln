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

import json
import os
from pathlib import Path
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from vllm.config import VllmConfig
else:
    VllmConfig = None
from vllm_rbln.logger import init_logger
from vllm_rbln.utils.optimum.registry import (get_rbln_model_info,
                                              is_enc_dec_arch, is_multi_modal,
                                              is_pooling_arch)

logger = init_logger(__name__)


def get_rbln_params(vllm_config: VllmConfig,
                    rbln_config: dict) -> tuple[int, int, int]:
    if is_enc_dec_arch(vllm_config.model_config.hf_config):
        max_seq_len = rbln_config.get("dec_max_seq_len")
        kvcache_block_size = max_seq_len
        batch_size = rbln_config.get("batch_size")
    elif is_multi_modal(vllm_config.model_config.hf_config):
        # Get configurations from main module (e.g. Qwen2.5-VL, Whisper)
        kvcache_block_size = rbln_config.get("kvcache_block_size")
        batch_size = rbln_config.get("batch_size")
        max_seq_len = rbln_config.get("max_seq_len")
        if max_seq_len is None:  # Whisper FIXME to be moved to enc-dec
            max_seq_len = rbln_config.get("dec_max_seq_len")
        # Get configurations from submodule
        if kvcache_block_size is None:
            submodules = ["language_model", "text_model"]
            for submodule in submodules:
                if submodule in rbln_config:
                    kvcache_block_size = rbln_config[submodule].get(
                        "kvcache_block_size", None)
                    batch_size = rbln_config[submodule].get("batch_size", None)
                    max_seq_len = rbln_config[submodule].get(
                        "max_seq_len", None)
                    if kvcache_block_size is not None:
                        break

    elif is_pooling_arch(vllm_config.model_config.hf_config):
        max_seq_len = rbln_config.get("max_seq_len")
        kvcache_block_size = max_seq_len
        batch_size = rbln_config.get("batch_size")
    else:
        # decoder
        kvcache_block_size = rbln_config.get("kvcache_block_size")
        batch_size = rbln_config.get("batch_size")
        max_seq_len = rbln_config.get("max_seq_len")

    assert kvcache_block_size is not None, (
        "kvcache_block_size must be specified in rbln_config.json")
    assert batch_size is not None, (
        "batch_size must be specified in rbln_config.json")
    assert max_seq_len is not None, (
        "max_seq_len must be specified in rbln_config.json")

    return kvcache_block_size, batch_size, max_seq_len


def update_vllm_config_with_rbln_params(vllm_config: VllmConfig,
                                        batch_size: int, max_model_len: int,
                                        kvcache_block_size: int) -> None:
    if vllm_config.scheduler_config.max_num_seqs != batch_size:
        logger.info(
            "Updating scheduler_config.max_num_seqs from %s to %s "
            "based on rbln_config.json",
            vllm_config.scheduler_config.max_num_seqs, batch_size)
        vllm_config.scheduler_config.max_num_seqs = batch_size

    if vllm_config.scheduler_config.max_num_batched_tokens != (max_model_len):
        logger.info(
            "Updating scheduler_config.max_num_batched_tokens from %s to "
            "%d based on rbln_config.json",
            vllm_config.scheduler_config.max_num_batched_tokens, max_model_len)
        vllm_config.scheduler_config.max_num_batched_tokens = (max_model_len)

    if vllm_config.model_config.max_model_len != max_model_len:
        logger.info(
            "Updating model_config.max_model_len and "
            "scheduler_config.max_model_len "
            "from %s to %s "
            "based on rbln_config.json",
            vllm_config.model_config.max_model_len, max_model_len)
        vllm_config.model_config.max_model_len = max_model_len
        vllm_config.scheduler_config.max_model_len = max_model_len

    if vllm_config.cache_config.enable_prefix_caching:
        if vllm_config.cache_config.block_size != 128:
            logger.info(
                "The block size is set to 128 for prefix caching in RBLN.")
        vllm_config.cache_config.block_size = 128
        if ("attn_block_size" in vllm_config.additional_config
                and vllm_config.additional_config["attn_block_size"]
                != kvcache_block_size):
            logger.info(
                "Updating attention block_size from %s to %s "
                "based on rbln_config.json",
                vllm_config.additional_config["attn_block_size"],
                kvcache_block_size)
        vllm_config.additional_config["attn_block_size"] = kvcache_block_size
    else:
        if vllm_config.cache_config.block_size != kvcache_block_size:
            logger.info(
                "Updating model_cache_config.block_size from %s to %s "
                "based on rbln_config.json",
                vllm_config.cache_config.block_size, kvcache_block_size)
            vllm_config.cache_config.block_size = kvcache_block_size


def is_qwen3_pooling(vllm_config: VllmConfig, ) -> bool:
    _, model_cls_name = get_rbln_model_info(vllm_config.model_config)
    return model_cls_name in ["RBLNQwen3ForCausalLM"
                              ] and vllm_config.model_config.task == "embed"


def get_rbln_config(vllm_config: VllmConfig) -> Optional[dict]:
    rbln_config_path = Path(
        os.path.join(vllm_config.model_config.model, "rbln_config.json"))
    if not rbln_config_path.exists():  # for pytest
        logger.warning(
            "rbln_config.json not found in model directory: %s. "
            "Using `block_size` from vllm_config.cache_config instead.",
            rbln_config_path)
        return None
    with open(rbln_config_path, encoding='utf-8') as f:
        rbln_config = json.load(f)
    return rbln_config


def sync_with_rbln_config(vllm_config: VllmConfig) -> None:
    try:
        rbln_config = get_rbln_config(vllm_config)
    except Exception as e:
        raise RuntimeError("Failed to get RBLN config: %s", e) from e

    if rbln_config is not None:
        kvcache_block_size, batch_size, max_model_len = \
            get_rbln_params(vllm_config, rbln_config)
        update_vllm_config_with_rbln_params(vllm_config, batch_size,
                                            max_model_len, kvcache_block_size)
