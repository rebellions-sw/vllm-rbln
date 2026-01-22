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





import torch


from vllm.config.kv_transfer import KVTransferConfig


@dataclass
class SendQueueItem:
    tensor_id: str
    remote_address: str
    tensor: torch.Tensor


class RBLNP2pGlooEngine:
    def __init__(self,
                 local_rank: int,
                 config: KVTransferConfig,
                 hostname: str = "",
                 port_offset: int = 0
    ) -> None:
        self.config = config
        self.rank = port_offset
        self.local_rank = local_rank

        self.device = torch.device("cpu")

        self.gpu_device = torch.device("rbln")

        self._hostname = "127.0.0.1"
        self._port = int(self.config.kv_port) + port_offset
        