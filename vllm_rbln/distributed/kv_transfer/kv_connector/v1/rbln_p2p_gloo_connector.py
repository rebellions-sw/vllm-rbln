# Copyright 2026 Rebellions Inc. All rights reserved.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at:

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Optional

import regex as re
import torch

from vllm.attention.backends.abstract import AttentionMetadata
from vllm.config import VllmConfig
from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorBase_V1,
    KVConnectorMetadata,
    KVConnectorRole,
)
from vllm.distributed.parallel_state import get_world_group
from vllm.logger import init_logger
from vllm.v1.attention.backends.mla.common import MLACommonMetadata
from vllm.v1.core.sched.output import SchedulerOutput

if TYPE_CHECKING:
    from vllm.forward_context import ForwardContext
    from vllm.v1.core.kv_cache_manager import KVCacheBlocks
    from vllm.v1.kv_cache_interface import KVCacheConfig
    from vllm.v1.request import Request

from vllm_rbln.distributed.kv_transfer.kv_connector.v1.rbln_p2p_gloo_engine import (
    RBLNP2pGlooEngine,
)

logger = init_logger(__name__)

@dataclass
class ReqMeta:
    # Request Id
    request_id: str
    # Request block ids
    block_ids: torch.Tensor
    # Request num tokens
    num_tokens: int

    @staticmethod
    def make_meta(
        request_id: str, token_ids: list[int], block_ids: list[int], block_size: int
    ) -> "ReqMeta":
        block_ids_tensor = torch.tensor(block_ids)
        return ReqMeta(
            request_id=request_id,
            block_ids=block_ids_tensor,
            num_tokens=len(token_ids),
        )


@dataclass
class RBLNP2pGlooConnectorMetadata(KVConnectorMetadata):
    requests: list[ReqMeta]

    def __init__(self):
        self.requests = []
    
    def add_request(
        self,
        request_id: str,
        token_ids: list[int],
        block_ids: list[int],
        block_size: int,
    ) -> None:
        self.requests.append(ReqMeta.make_meta(request_id, token_ids, block_ids, block_size))


class RBLNP2pGlooConnector(KVConnectorBase_V1):
    """ RBLN P2P KV cache connector using Gloo.

    This connector is used to transfer KV cache between workers using Gloo.
    """

    def __init__(
        self,
        vllm_config: "VllmConfig",
        role: KVConnectorRole,
        kv_cache_config: Optional["KVCacheConfig"] = None,
    ):
        super().__init__(
            vllm_config=vllm_config,
            role=role,
        )
        self._block_size = vllm_config.cache_config.block_size
        self._requests_need_load: dict[str, Any] = {}
        self.is_producer = self._kv_transfer_config.is_kv_producer
        self.chunked_prefill: dict[str, tuple[list[int], list[int] | None]] = {}

        self._rank = get_world_group().rank if role == KVConnectorRole.WORKER else 0
        self._local_rank = (
            get_world_group().rank if role == KVConnectorRole.WORKER else 0
        )

        self.rbln_p2p_gloo_engine = (
            RBLNP2pGlooEngine(
                local_rank=self._local_rank,
                config=self._kv_transfer_config,
                hostname="",
                port_offset=self._rank,
            )
            if role == KVConnectorRole.WORKER
            else None
        )
    
    # ==============================
    # Worker-side methods
    # ==============================

    def start_load_kv()


    def build_connector_meta(
        self,
        scheduler_output: SchedulerOutput,
    ) -> RBLNP2pGlooConnectorMetadata:
        """Build the connector metadata for this step.

        This function should NOT modify any fields in the scheduler_output.
        Also, calling this function will reset the state of the connector.

        Args:
            scheduler_output (SchedulerOutput): the scheduler output object.
        """

        meta = RBLNP2pGlooConnectorMetadata()

        # Simplified version: only handle new requests
        # No preemption, no continuous batching complexity
        for new_req in scheduler_output.scheduled_new_reqs:
            if self.is_producer:
                # Producer: save all prefilled requests
                meta.add_request(request_id=new_req.req_id,
                                 token_ids=new_req.prompt_token_ids,
                                 block_ids=new_req.block_ids[0],
                                 block_size=self._block_size)
            else:
                # Consumer: load requests that need external KV
                if new_req.req_id in self._requests_need_load:
                    meta.add_request(request_id=new_req.req_id,
                                     token_ids=new_req.prompt_token_ids,
                                     block_ids=new_req.block_ids[0],
                                     block_size=self._block_size)
                    self._requests_need_load.pop(new_req.req_id)

        self._requests_need_load.clear()
        return meta