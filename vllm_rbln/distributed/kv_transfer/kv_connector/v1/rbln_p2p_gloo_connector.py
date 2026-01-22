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
import safetensors
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

# from vllm_rbln.distributed.kv_transfer.kv_connector.v1.rbln_p2p_gloo_engine import (
#     RBLNP2pGlooEngine,
# )

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

    def __init__(self, vllm_config: "VllmConfig", role: KVConnectorRole):
        super().__init__(vllm_config=vllm_config, role=role)
        self._block_size = vllm_config.cache_config.block_size
        self._requests_need_load: dict[str, Any] = {}
        self.config = vllm_config.kv_transfer_config
        self.is_producer = self.config.is_kv_producer

        self._rank = get_world_group().rank if role == KVConnectorRole.WORKER else 0
        self._local_rank = get_world_group().local_rank if role == KVConnectorRole.WORKER else 0

        # self.rbln_p2p_gloo_engine = RBLNP2pGlooEngine(
        #     local_rank=self._local_rank,
        #     config=self.config,
        #     hostname="",
        #     port_offset=self._rank,
        # ) if role == KVConnectorRole.WORKER else None
    
    # ==============================
    # Worker-side methods
    # ==============================

    def start_load_kv(self, forward_context: "ForwardContext", **kwargs) -> None:
        """Start loading the KV cache from the connector buffer to vLLM's paged KV buffer.

        Args:
            forward_context (ForwardContext): the forward context.
            **kwargs: additional arguments for the load operation.
        """

        logger.info("****************************** start_load_kv")

        # Only consumer/decode loads KV Cache
        if self.is_producer:
            return
        
        # assert self.rbln_p2p_gloo_engine is not None
        logger.info("****************************** self.is_producer: %s", self.is_producer)

        attn_metadata = forward_context.attn_metadata
        if attn_metadata is None:
            return

        # def inject_kv_into_layer(
        #     layer: torch.Tensor,
        #     kv_cache: torch.Tensor,
        #     block_ids: torch.Tensor,
        #     request_id: str,
        # ) -> None:
        #     """
        #     Inject KV cache data into a given attention layer tensor.

        #     This function updates `layer` in-place with values from `kv_cache`,
        #     handling different backend layouts:
        #       - MLA (Multi-linear Attention) or FlashInfer: KV tensors are
        #         indexed along the first dimension.
        #       - FlashAttention: KV tensors are indexed along the second
        #         dimension.
        #     """
        #     if (isinstance(attn_metadata, MLACommonMetadata)
        #             or layer.shape[1] == 2): # MLA or FlashInfer
        #         num_block = kv_cache.shape[0]
        #         self.check_tensors_except_dim(layer, kv_cache, 0)
        #         if len(block_ids) == num_block:
        #             layer[block_ids, ...] = kv_cache
        #         else:
        #             layer[block_ids[:num_block], ...] = kv_cache
        #             logger.warning(
        #                 "🚧kv_cache does not match, block_ids:%d, "
        #                 "num_block:%d, request_id:%s", len(block_ids),
        #                 num_block, request_id)
                
        #     elif layer.shape[0] == 2: # FlashAttention
        #         num_block = kv_cache.shape[1]
        #         self.check_tensors_except_dim(layer, kv_cache, 1)
        #         if len(block_ids) == num_block:
        #             layer[:, block_ids, ...] = kv_cache
        #         else:
        #             layer[:, block_ids[:num_block], ...] = kv_cache
        #             logger.warning(
        #                 "🚧kv_cache does not match, block_ids:%d, "
        #                 "num_block:%d, request_id:%s", len(block_ids),
        #                 num_block, request_id)

        def inject_kv_into_layer(
            dst_kv_cache_layer: torch.Tensor,
            src_kv_cache: torch.Tensor,
            slot_mapping: torch.Tensor,
        ) -> None:
            """Inject the KV cache into the layer.

            Args:
                dst_kv_cache_layer (torch.Tensor): the destination KV cache
                    layer. In shape [2, num_pages, page_size, xxx] if not
                    using MLA, [num_pages, page_size, xxx] otherwise.
                src_kv_cache (torch.Tensor): the source KV cache. In shape
                    [2, num_tokens, xxx] if not using MLA, [num_tokens, xxx]
                    otherwise.
                slot_mapping (torch.Tensor): the slot mapping. In shape
                    [num_tokens].
            """
            dst_kv_cache_layer_shape = dst_kv_cache_layer.shape
            if isinstance(attn_metadata, MLACommonMetadata):
                num_pages = dst_kv_cache_layer_shape[0]
                page_size = dst_kv_cache_layer_shape[1]
                dst_kv_cache_layer = dst_kv_cache_layer.reshape(
                    num_pages * page_size, -1
                )
                dst_kv_cache_layer[slot_mapping, ...] = src_kv_cache
                dst_kv_cache_layer.reshape(dst_kv_cache_layer_shape)
            else:
                num_pages = dst_kv_cache_layer_shape[1]
                page_size = dst_kv_cache_layer_shape[2]
                dst_kv_cache_layer = dst_kv_cache_layer.reshape(
                    2, num_pages * page_size, -1
                )
                dst_kv_cache_layer[:, slot_mapping, ...] = src_kv_cache
                dst_kv_cache_layer.reshape(dst_kv_cache_layer_shape)
        
        # Get the metadata
        metadata: KVConnectorMetadata = self._get_connector_metadata()
        assert isinstance(metadata, RBLNP2pGlooConnectorMetadata)

        if metadata is None:
            return
        
        # Load the KV for each request each layer
        logger.info("****************************** metadata.requests: %s", metadata.requests)
        for request in metadata.requests:
            logger.info("****************************** request: %s", request)
            for layer_name in forward_context.no_compile_layers:
                logger.info("****************************** layer_name: %s", layer_name)
                layer = forward_context.no_compile_layers[layer_name]

                # Only process layers that have kv_cache
                # attributed (attention layers) Skip non-attention
                # layers like FusedMoE
                kv_cache = getattr(layer, "kv_cache", None)
                if kv_cache is None:
                    continue

                layer = kv_cache[forward_context.virtual_engine]

                # kv_cache = self.rbln_p2p_gloo_engine.recv_tensor(
                #     request.request_id + "#" + layer_name)

                # if kv_cache is None:
                #     logger.warning("🚧kv_cache is None, %s", request.request_id)
                #     continue

                # inject_kv_into_layer(layer, kv_cache, request.block_ids,
                #                      request.request_id)

                filename = self._generate_filename_debug(
                    layer_name, request.token_ids, request.mm_hashes
                )
                kv_cache = safetensors.torch.load_file(filename)["kv_cache"].cuda()
                inject_kv_into_layer(layer, kv_cache, request.slot_mapping)
    
    def wait_for_layer_load(self, layer_name: str) -> None:
        return

    def save_kv_layer(self, layer_name: str, kv_layer: torch.Tensor,
                      attn_metadata: "AttentionMetadata", **kwargs) -> None:
        """Start saving the KV cache of the layer from vLLM's paged buffer
        to the connector.

        Args:
            layer_name (str): the name of the layer.
            kv_layer (torch.Tensor): the paged KV buffer of the current
                layer in vLLM.
            attn_metadata (AttentionMetadata): the attention metadata.
            **kwargs: additional arguments for the save operation.
        """

        # Only producer/prefill saves KV Cache
        if not self.is_producer:
            return
        
        # assert self.rbln_p2p_gloo_engine is not None

        # def extract_kv_from_layer(
        #     layer: torch.Tensor,
        #     block_ids: torch.Tensor,
        # ) -> torch.Tensor:
        #     """
        #     Extract KV cache slices from a given attention layer tensor.

        #     This function handles multiple backend layouts:
        #       - MLA (Multi-Linear Attention) or FlashInfer: KV tensors are
        #         indexed along the first dimension.
        #       - FlashAttention: KV tensors are indexed along the second
        #         dimension.
        #     """
        #     if (isinstance(attn_metadata, MLACommonMetadata)
        #             or layer.shape[1] == 2):  # MLA or FlashInfer
        #         return layer[block_ids, ...]

        #     if layer.shape[0] == 2:  # FlashAttention
        #         return layer[:, block_ids, ...]

        #     return None


        def extract_kv_from_layer(
            layer: torch.Tensor,
            slot_mapping: torch.Tensor,
        ) -> torch.Tensor:
            """Extract the KV cache from the layer.

            Assume the shape of the layer is (2, num_pages, page_size, xxx)
            if MLA is not used, and (num_pages, page_size, xxx) otherwise.
            """
            if isinstance(attn_metadata, MLACommonMetadata):
                num_pages, page_size = layer.shape[0], layer.shape[1]
                return layer.reshape(num_pages * page_size, -1)[slot_mapping, ...]
            num_pages, page_size = layer.shape[1], layer.shape[2]
            return layer.reshape(2, num_pages * page_size, -1)[:, slot_mapping, ...]

        connector_metadata = self._get_connector_metadata()
        assert isinstance(connector_metadata, RBLNP2pGlooConnectorMetadata)
        for request in connector_metadata.requests:
            # request_id = request.request_id
            # ip, port = self.parse_request_id(request_id, True)
            # remote_address = ip + ":" + str(port + self._rank)

            # kv_cache = extract_kv_from_layer(kv_layer, request.block_ids)
            # self.rbln_p2p_gloo_engine.send_tensor(request_id + "#" + layer_name, kv_cache, remote_address)
            if request.is_store:
                filename = self._generate_filename_debug(
                    layer_name, request.token_ids, request.mm_hashes
                )
                kv_cache = extract_kv_from_layer(kv_layer, request.slot_mapping)
                tensors = {"kv_cache": kv_cache.detach().cpu()}
                safetensors.torch.save_file(tensors, filename)

    def wait_for_save(self):
        return

    # ==============================
    # Scheduler-side methods
    # ==============================

    def get_num_new_matched_tokens(
        self,
        request: "Request",
        num_computed_tokens: int,
    ) -> tuple[int, bool]:
        """
        Get number of new tokens that can be loaded from the
        external KV cache beyond the num_computed_tokens.

        Args:
            request (Request): the request object.
            num_computed_tokens (int): the number of locally
                computed tokens for this request
        
        Returns:
            the number of tokens that can be loaded from the
            external KV cache beyond what is already computed.
        """
        if self.is_producer:
            return 0, False
        
        num_external_tokens = (len(request.prompt_token_ids) - 1 - num_computed_tokens)

        if num_external_tokens < 0:
            num_external_tokens = 0
        
        return num_external_tokens, False
    
    def update_state_after_alloc(self, request: "Request",
                                 blocks: "KVCacheBlocks",
                                 num_external_tokens: int):
        """
        Update KVConnector state after block allocation.
        """
        if not self.is_producer and num_external_tokens > 0:
            self._requests_need_load[request.request_id] = (
                request, blocks.get_block_ids()[0])

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
    

    # ==============================
    # Static methods
    # ==============================

    @staticmethod
    def parse_request_id(request_id: str, is_prefill=True) -> tuple[str, int]:
        ip = "127.0.0.1"
        port = 12345
        if is_prefill:
            return ip, port
        else:
            return ip, port + 1
    
    @staticmethod
    def check_tensors_except_dim(tensor1, tensor2, dim):
        shape1 = tensor1.size()
        shape2 = tensor2.size()

        if len(shape1) != len(shape2) or not all(
                s1 == s2
                for i, (s1, s2) in enumerate(zip(shape1, shape2)) if i != dim):
            raise NotImplementedError(
                "Currently, only symmetric TP is suppoerted. Asymmetric TP, PP,"
                "and others will be supported in the future PRs.")
    
    def _generate_foldername_debug(
        self,
        token_ids: torch.Tensor,
        mm_hashes: list[str],
        create_folder=False,
    ) -> str:
        """Generate a folder name based on the hash of the bytes of the input
        ids.
        """
        token_bytes = token_ids.numpy().tobytes()
        # Add mm_hashes to the bytes being hashed to avoid path traversal and
        # to create a canonical key.
        if mm_hashes:
            mm_str = "-".join(mm_hashes)
            token_bytes += mm_str.encode("utf-8")
        input_ids_hash = safe_hash(token_bytes, usedforsecurity=False).hexdigest()

        foldername = os.path.join(self._storage_path, input_ids_hash)
        if create_folder:
            os.makedirs(foldername, exist_ok=True)
        return foldername

    def _generate_filename_debug(
        self,
        layer_name: str,
        token_ids: torch.Tensor,
        mm_hashes: list[str],
    ) -> str:
        """Generate a file name based on the layer name and the hash
        of the bytes of the input ids.
        """
        foldername = self._generate_foldername_debug(
            token_ids, mm_hashes=mm_hashes, create_folder=True
        )
        return os.path.join(foldername, f"{layer_name}.safetensors")