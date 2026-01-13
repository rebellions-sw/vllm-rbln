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

# Copied from vllm.v1.core.sched.async_scheduler: https://github.com/vllm-project/vllm/blob/v0.10.2/vllm/v1/core/sched/async_scheduler.py
# The only differences are:
# - Inherit RBLNScheduler instead of Scheduler
# - Use vllm-rbln logger

from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.request import Request, RequestStatus

from vllm_rbln.logger import init_logger
from vllm_rbln.v1.core.rbln_scheduler import RBLNScheduler

logger = init_logger(__name__)


class RBLNAsyncScheduler(RBLNScheduler):

    def _update_after_schedule(
        self,
        scheduler_output: SchedulerOutput,
    ) -> None:
        super()._update_after_schedule(scheduler_output)
        pending_structured_output_tokens = False
        spec_decode_tokens = scheduler_output.scheduled_spec_decode_tokens
        for req_id in scheduler_output.num_scheduled_tokens:
            request = self.requests[req_id]
            pending_structured_output_tokens |= (
                request.use_structured_output
                and request.num_output_placeholders > 0)
            cur_num_spec_tokens = len(spec_decode_tokens.get(req_id, ()))
            if (request.num_computed_tokens == request.num_tokens +
                    request.num_output_placeholders + cur_num_spec_tokens):
                # The request will generate a new token plus num_spec_tokens
                # in this scheduling step.
                request.num_output_placeholders += 1 + cur_num_spec_tokens
                # Add placeholders for the new tokens in spec_token_ids.
                # We will update the actual spec token ids in the worker
                # process.
                request.spec_token_ids = [-1] * self.num_spec_tokens

        scheduler_output.pending_structured_output_tokens = (
            pending_structured_output_tokens)

    def _update_request_with_output(
        self,
        request: Request,
        new_token_ids: list[int],
    ) -> tuple[list[int], bool]:
        if request.discard_latest_async_tokens:
            # If the request is force preempted in reset_prefix_cache, we
            # should discard the latest async token.
            request.discard_latest_async_tokens = False
            return [], False

        status_before_update = request.status
        new_token_ids, stopped = super()._update_request_with_output(
            request, new_token_ids)

        # Update the number of output placeholders.
        request.num_output_placeholders -= len(new_token_ids)
        assert request.num_output_placeholders >= 0

        # Cache the new tokens. Preempted requests should be skipped.
        if status_before_update == RequestStatus.RUNNING:
            self.kv_cache_manager.cache_blocks(
                request,
                request.num_computed_tokens - request.num_output_placeholders)
        return new_token_ids, stopped
