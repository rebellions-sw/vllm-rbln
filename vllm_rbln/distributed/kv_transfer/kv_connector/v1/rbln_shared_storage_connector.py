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

from __future__ import annotations

import hashlib
import os
import uuid
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import torch
from vllm.attention.backends.abstract import AttentionMetadata
from vllm.distributed.kv_transfer.kv_connector.v1 import (KVConnectorBase_V1,
                                                          KVConnectorRole)
from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorMetadata)
from vllm.logger import init_logger

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.forward_context import ForwardContext
    from vllm.v1.core.kv_cache_manager import KVCacheBlocks
    from vllm.v1.core.sched.output import SchedulerOutput
    from vllm.v1.kv_cache_interface import KVCacheConfig
    from vllm.v1.request import Request

logger = init_logger(__name__)


@dataclass
class RBLNReqMeta:
    token_ids: torch.Tensor
    slot_mapping: torch.Tensor
    is_store: bool

    @staticmethod
    def make_meta(
        token_ids: list[int],
        block_ids: list[int],
        block_size: int,
        is_store: bool,
    ) -> RBLNReqMeta:
        valid_num_tokens = (len(token_ids) // block_size) * block_size
        token_ids_tensor = torch.tensor(token_ids,
                                        dtype=torch.long)[:valid_num_tokens]
        block_ids_tensor = torch.tensor(block_ids, dtype=torch.long)
        num_blocks = block_ids_tensor.shape[0]
        block_offsets = torch.arange(0, block_size, dtype=torch.long)
        slot_mapping = (block_offsets.reshape(
            (1, block_size)) + block_ids_tensor.reshape(
                (num_blocks, 1)) * block_size)
        slot_mapping = slot_mapping.flatten()[:valid_num_tokens]
        return RBLNReqMeta(
            token_ids=token_ids_tensor,
            slot_mapping=slot_mapping,
            is_store=is_store,
        )


@dataclass
class RBLNConnectorMetadata(KVConnectorMetadata):
    requests: list[RBLNReqMeta] = field(default_factory=list)


class RBLNSharedStorageConnector(KVConnectorBase_V1):
    """Shared-storage KV connector for RBLN PD disaggregation."""

    def __init__(
        self,
        vllm_config: VllmConfig,
        role: KVConnectorRole,
        kv_cache_config: KVCacheConfig | None = None,
    ) -> None:
        super().__init__(vllm_config=vllm_config,
                         role=role,
                         kv_cache_config=kv_cache_config)
        self._block_size = vllm_config.cache_config.block_size
        self._requests_need_load: dict[str, Request] = {}
        self._requests_need_load_block_ids: dict[str, list[int]] = {}
        self._storage_path = self._kv_transfer_config.get_from_extra_config(
            "shared_storage_path", "/tmp/rbln_kv_cache")
        self.is_producer = self._kv_transfer_config.is_kv_producer
        self._kv_caches: dict[str, torch.Tensor] | None = None
        self._layer_names: set[str] | None = None
        self._saved_layers_by_req_hash: dict[str, set[str]] = {}
        os.makedirs(self._storage_path, exist_ok=True)

    # ==============================
    # Worker-side methods
    # ==============================

    def register_kv_caches(self, kv_caches: dict[str, torch.Tensor]):
        self._kv_caches = kv_caches
        self._layer_names = set(kv_caches.keys())

    def start_load_kv(self, forward_context: ForwardContext,
                      **kwargs: Any) -> None:
        if self.is_producer:
            return
        if not self.has_connector_metadata():
            return
        if self._kv_caches is None:
            logger.warning("KV caches are not registered for load")
            return
        metadata = self._get_connector_metadata()
        assert isinstance(metadata, RBLNConnectorMetadata)
        for req_meta in metadata.requests:
            if req_meta.is_store:
                continue
            request_hash = self._get_request_hash(req_meta.token_ids)
            for layer_name, kv_layer in self._kv_caches.items():
                layer_path = self._get_layer_path(request_hash, layer_name)
                if not os.path.exists(layer_path):
                    logger.warning(
                        "Missing KV layer file for load: %s",
                        layer_path,
                    )
                    continue
                src_kv = torch.load(layer_path, map_location="cpu")
                if kv_layer.device.type != "cpu":
                    logger.warning("KV cache is not on CPU for RBLN load")
                self._inject_kv(kv_layer, src_kv, req_meta.slot_mapping)

    def wait_for_layer_load(self, layer_name: str) -> None:
        return

    def save_kv_layer(
        self,
        layer_name: str,
        kv_layer: torch.Tensor,
        attn_metadata: AttentionMetadata,
        **kwargs: Any,
    ) -> None:
        if not self.is_producer:
            return
        if not self.has_connector_metadata():
            return
        metadata = self._get_connector_metadata()
        assert isinstance(metadata, RBLNConnectorMetadata)
        for req_meta in metadata.requests:
            if not req_meta.is_store:
                continue
            request_hash = self._get_request_hash(req_meta.token_ids)
            kv_to_save = self._extract_kv(kv_layer, req_meta.slot_mapping)
            self._save_layer(request_hash, layer_name, kv_to_save)
            self._mark_layer_saved(request_hash, layer_name)

    def wait_for_save(self):
        return

    # ==============================
    # Scheduler-side methods
    # ==============================

    def get_num_new_matched_tokens(
        self,
        request: Request,
        num_computed_tokens: int,
    ) -> tuple[int | None, bool]:
        if request.prompt_token_ids is None:
            return 0, False
        aligned_tokens = (len(request.prompt_token_ids) //
                          self._block_size) * self._block_size
        if aligned_tokens <= 0:
            return 0, False
        # Hash only the block-aligned prefix (must match worker
        # side which hashes token_ids[:valid_num_tokens]).
        token_ids = torch.tensor(request.prompt_token_ids,
                                 dtype=torch.long)[:aligned_tokens]
        request_hash = self._get_request_hash(token_ids)
        if not os.path.exists(self._get_ready_path(request_hash)):
            return 0, False
        num_new_tokens = aligned_tokens - num_computed_tokens
        if num_new_tokens <= 0:
            return 0, False
        return num_new_tokens, False

    def update_state_after_alloc(
        self,
        request: Request,
        blocks: KVCacheBlocks,
        num_external_tokens: int,
    ) -> None:
        if num_external_tokens <= 0:
            return
        block_groups = blocks.get_block_ids(allow_none=True)
        if block_groups is None or not block_groups:
            return
        self._requests_need_load[request.request_id] = request
        self._requests_need_load_block_ids[request.request_id] = list(
            block_groups[0])

    def build_connector_meta(
            self, scheduler_output: SchedulerOutput) -> KVConnectorMetadata:
        metadata = RBLNConnectorMetadata()
        if self.is_producer:
            for new_req in scheduler_output.scheduled_new_reqs:
                if new_req.prompt_token_ids is None:
                    continue
                if not new_req.block_ids:
                    continue
                metadata.requests.append(
                    RBLNReqMeta.make_meta(
                        new_req.prompt_token_ids,
                        list(new_req.block_ids[0]),
                        self._block_size,
                        is_store=True,
                    ))
        else:
            for req_id, request in self._requests_need_load.items():
                block_ids = self._requests_need_load_block_ids.get(req_id)
                if request.prompt_token_ids is None or not block_ids:
                    continue
                metadata.requests.append(
                    RBLNReqMeta.make_meta(
                        request.prompt_token_ids,
                        block_ids,
                        self._block_size,
                        is_store=False,
                    ))
            self._requests_need_load.clear()
            self._requests_need_load_block_ids.clear()
        return metadata

    # ==============================
    # Storage helpers
    # ==============================

    def _get_request_hash(self, token_ids: torch.Tensor) -> str:
        token_bytes = token_ids.detach().cpu().numpy().tobytes()
        return hashlib.sha256(token_bytes).hexdigest()

    def _get_request_dir(self, request_hash: str) -> str:
        return os.path.join(self._storage_path, request_hash)

    def _get_layer_path(self, request_hash: str, layer_name: str) -> str:
        return os.path.join(self._get_request_dir(request_hash),
                            f"{layer_name}.pt")

    def _get_ready_path(self, request_hash: str) -> str:
        return os.path.join(self._get_request_dir(request_hash), "_READY")

    def _save_layer(self, request_hash: str, layer_name: str,
                    kv_tensor: torch.Tensor) -> None:
        request_dir = self._get_request_dir(request_hash)
        kv_tensor = kv_tensor.detach().cpu()
        if not os.path.exists(request_dir):
            tmp_dir = f"{request_dir}.tmp.{uuid.uuid4().hex}"
            os.makedirs(tmp_dir, exist_ok=True)
            torch.save(kv_tensor, os.path.join(tmp_dir, f"{layer_name}.pt"))
            os.rename(tmp_dir, request_dir)
            return
        os.makedirs(request_dir, exist_ok=True)
        tmp_path = os.path.join(request_dir,
                                f".{layer_name}.tmp.{uuid.uuid4().hex}.pt")
        torch.save(kv_tensor, tmp_path)
        os.rename(tmp_path, self._get_layer_path(request_hash, layer_name))

    def _write_ready_marker(self, request_hash: str) -> None:
        request_dir = self._get_request_dir(request_hash)
        os.makedirs(request_dir, exist_ok=True)
        ready_path = self._get_ready_path(request_hash)
        if os.path.exists(ready_path):
            return
        tmp_path = os.path.join(request_dir, f"._READY.tmp.{uuid.uuid4().hex}")
        with open(tmp_path, "w", encoding="utf-8"):
            pass
        os.rename(tmp_path, ready_path)

    def _mark_layer_saved(self, request_hash: str, layer_name: str) -> None:
        if self._layer_names is None:
            self._write_ready_marker(request_hash)
            return
        saved_layers = self._saved_layers_by_req_hash.setdefault(
            request_hash, set())
        saved_layers.add(layer_name)
        if saved_layers >= self._layer_names:
            self._write_ready_marker(request_hash)
            self._saved_layers_by_req_hash.pop(request_hash, None)

    @staticmethod
    def _extract_kv(kv_layer: torch.Tensor,
                    slot_mapping: torch.Tensor) -> torch.Tensor:
        """Extract KV data from RBLN paged buffer by slot_mapping."""
        shape = kv_layer.shape
        num_blocks = shape[1]
        block_size = shape[4]
        kv = kv_layer.squeeze(3)
        kv = kv.permute(0, 1, 3, 2, 4)
        kv = kv.contiguous().reshape(2, num_blocks * block_size, -1)
        return kv[:, slot_mapping, :]

    @staticmethod
    def _inject_kv(
        kv_layer: torch.Tensor,
        src_kv: torch.Tensor,
        slot_mapping: torch.Tensor,
    ) -> None:
        """Inject KV data into RBLN paged buffer by slot_mapping."""
        shape = kv_layer.shape
        num_blocks, num_kv_heads, block_size, head_size = (
            shape[1],
            shape[2],
            shape[4],
            shape[5],
        )
        kv_flat = kv_layer.squeeze(3).permute(0, 1, 3, 2, 4).contiguous()
        kv_flat = kv_flat.reshape(2, num_blocks * block_size, -1)
        if src_kv.device != kv_flat.device:
            src_kv = src_kv.to(kv_flat.device)
        kv_flat[:, slot_mapping, :] = src_kv
        kv_reshaped = kv_flat.reshape(
            2,
            num_blocks,
            block_size,
            num_kv_heads,
            head_size,
        )
        kv_reshaped = kv_reshaped.permute(0, 1, 3, 2, 4).unsqueeze(3)
        kv_layer.copy_(kv_reshaped)
