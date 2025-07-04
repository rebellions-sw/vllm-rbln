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

import random
import time
from collections import deque
from dataclasses import dataclass
from typing import Deque, List, Optional, Set, Tuple

from vllm.config import SchedulerConfig
from vllm.core.interfaces import AllocStatus
from vllm.core.scheduler import (ARTIFICIAL_PREEMPTION_PROB,
                                 PartialPrefillMetadata, PreemptionMode,
                                 ScheduledSequenceGroup, Scheduler,
                                 SchedulerOutputs, SchedulerPrefillOutputs,
                                 SchedulerRunningOutputs,
                                 SchedulerSwappedInOutputs, SchedulingBudget)
from vllm.sequence import SequenceGroup, SequenceStage, SequenceStatus

from vllm_rbln.logger import init_logger

logger = init_logger(__name__)


class RBLNScheduler(Scheduler):

    def _schedule_running(
        self,
        budget: SchedulingBudget,
        curr_loras: Optional[Set[int]],
        enable_chunking: bool = False,
        partial_prefill_metadata: Optional[PartialPrefillMetadata] = None,
    ) -> SchedulerRunningOutputs:
        """Schedule sequence groups that are running.

        Running queue should include decode and chunked prefill requests.

        Args:
            budget: The scheduling budget. The argument is in-place updated
                when any decodes are preempted.
            curr_loras: Currently batched lora request ids. The argument is
                in-place updated when any decodes are preempted.
            enable_chunking: If True, seq group can be chunked and only a
                chunked number of tokens are scheduled  if
                `budget.num_batched_tokens` has not enough capacity to schedule
                all tokens.
            partial_prefill_metadata: information about the partial prefills
            that are currently running

        Returns:
            SchedulerRunningOutputs.
        """
        ret: SchedulerRunningOutputs = self._scheduler_running_outputs_cache[
            self.cache_id].get_object()
        ret.blocks_to_swap_out.clear()
        ret.blocks_to_copy.clear()
        ret.decode_seq_groups.clear()
        ret.prefill_seq_groups.clear()
        ret.preempted.clear()
        ret.swapped_out.clear()

        ret.num_lookahead_slots = self._get_num_lookahead_slots(
            is_prefill=False, enable_chunking=enable_chunking)

        ret.decode_seq_groups_list.clear()
        ret.prefill_seq_groups_list.clear()

        # Blocks that need to be swapped or copied before model execution.
        blocks_to_swap_out: List[Tuple[int, int]] = ret.blocks_to_swap_out
        blocks_to_copy: List[Tuple[int, int]] = ret.blocks_to_copy

        decode_seq_groups: List[ScheduledSequenceGroup] = ret.decode_seq_groups
        prefill_seq_groups: List[
            ScheduledSequenceGroup] = ret.prefill_seq_groups
        preempted: List[SequenceGroup] = ret.preempted
        swapped_out: List[SequenceGroup] = ret.swapped_out

        running_queue = self.running
        assert len(self._async_stopped) == 0
        while running_queue:
            seq_group = running_queue[0]
            # We discard the cached tokens info here because we don't need it
            # for running sequence:
            #   1. If a sequence is running with chunked prefill, the cached
            #      tokens info was already used for the first prefill.
            #   2. If a sequence is running with non-chunked prefill, then
            #      there it's a decoding sequence, and the cached tokens info is
            #      irrelevant.
            num_uncached_new_tokens, _ = \
                self._get_num_new_uncached_and_cached_tokens(
                    seq_group,
                    SequenceStatus.RUNNING,
                    enable_chunking,
                    budget,
                    partial_prefill_metadata,
                )

            num_running_tokens = num_uncached_new_tokens
            if num_running_tokens == 0:
                # No budget => Stop
                break

            running_queue.popleft()

            # With async postprocessor, an extra decode run is done
            # to process the final tokens. The check below avoids this extra
            # decode run when the model max len is reached, in order to avoid
            # a memory overflow.
            if self.use_async_output_proc and seq_group.seqs[0].get_len(
            ) > self.scheduler_config.max_model_len:
                self._async_stopped.append(seq_group)
                continue

            # NOTE(woosuk): Preemption happens only when there is no available
            # slot to keep all the sequence groups in the RUNNING state.
            while not self._can_append_slots(seq_group, enable_chunking):
                budget.subtract_num_batched_tokens(seq_group.request_id,
                                                   num_running_tokens)
                num_running_seqs = seq_group.get_max_num_running_seqs()
                budget.subtract_num_seqs(seq_group.request_id,
                                         num_running_seqs)

                if curr_loras is not None and \
                    seq_group.lora_int_id > 0 and \
                    seq_group.lora_int_id in curr_loras:
                    curr_loras.remove(seq_group.lora_int_id)

                # Determine victim sequence
                cont_loop = True
                if running_queue:
                    # Preempt the lowest-priority sequence group.
                    victim_seq_group = running_queue.pop()
                else:
                    # No other sequence group can be preempted.
                    # Preempt the current sequence group.
                    # Note: This is also where we stop this loop
                    # (since there is nothing else to preempt)
                    victim_seq_group = seq_group
                    cont_loop = False

                # With async postprocessor, before preempting a sequence
                # we need to ensure it has no pending async postprocessor
                do_preempt = True
                if self.use_async_output_proc:
                    assert self.output_proc_callback is not None
                    self.output_proc_callback(
                        request_id=victim_seq_group.request_id)

                    # It may be that the async pending "victim_seq_group"
                    # becomes finished, in which case we simply free it.
                    if victim_seq_group.is_finished():
                        self._free_finished_seq_group(victim_seq_group)
                        do_preempt = False

                # Do preemption
                if do_preempt:
                    preempted_mode = self._preempt(victim_seq_group,
                                                   blocks_to_swap_out)
                    if preempted_mode == PreemptionMode.RECOMPUTE:
                        preempted.append(victim_seq_group)
                    else:
                        swapped_out.append(victim_seq_group)

                if not cont_loop:
                    break
            else:
                self._append_slots(seq_group, blocks_to_copy, enable_chunking)
                is_prefill = seq_group.is_prefill()

                scheduled_seq_group: ScheduledSequenceGroup = \
                    self._scheduled_seq_group_cache[
                    self.cache_id].get_object()
                scheduled_seq_group.seq_group = seq_group
                if is_prefill:
                    scheduled_seq_group.token_chunk_size = num_running_tokens
                    prefill_seq_groups.append(scheduled_seq_group)
                    ret.prefill_seq_groups_list.append(seq_group)
                else:
                    scheduled_seq_group.token_chunk_size = 1
                    decode_seq_groups.append(scheduled_seq_group)
                    ret.decode_seq_groups_list.append(seq_group)

                budget.add_num_batched_tokens(seq_group.request_id,
                                              num_running_tokens)
                # OPTIMIZATION:  Note that get_max_num_running_seqs is
                # expensive. For the default scheduling chase where
                # enable_chunking is False, num_seqs are updated before running
                # this method, so we don't have to update it again here.
                if enable_chunking:
                    num_running_seqs = seq_group.get_max_num_running_seqs()
                    budget.add_num_seqs(seq_group.request_id, num_running_seqs)
                if curr_loras is not None and seq_group.lora_int_id > 0:
                    curr_loras.add(seq_group.lora_int_id)

            # NOTE(jiwoo.park): RBLN allows only a single prefill.
            # Also, it can't schedule with decode.
            if len(ret.prefill_seq_groups_list) > 0:
                break

        self._scheduler_running_outputs_cache[self.next_cache_id].reset()
        self._scheduled_seq_group_cache[self.next_cache_id].reset()

        return ret

    def _schedule_prefills(
        self,
        budget: SchedulingBudget,
        curr_loras: Optional[Set[int]],
        enable_chunking: bool = False,
        partial_prefill_metadata: Optional[PartialPrefillMetadata] = None,
    ) -> SchedulerPrefillOutputs:
        """Schedule sequence groups that are in prefill stage.

        Note that the current scheduler treats PREEMPTED_FOR_RECOMPUTE
        as a new prefill (that starts from beginning -> most recently generated
        tokens).

        It schedules waiting requests as long as it fits `budget` and
        curr_loras <= max_lora from the scheduling config. The input arguments
        `budget` and `curr_loras` are updated based on scheduled seq_groups.

        Args:
            budget: The scheduling budget. The argument is in-place updated
                when any requests are scheduled.
            curr_loras: Currently batched lora request ids. The argument is
                in-place updated when any requests are scheduled.
            enable_chunking: If True, seq group can be chunked and only a
                chunked number of tokens are scheduled  if
                `budget.num_batched_tokens` has not enough capacity to schedule
                all tokens.
            partial_prefill_metadata: information about the partial prefills
                that are currently running

        Returns:
            SchedulerPrefillOutputs.
        """
        if budget.remaining_token_budget() == 0:
            # Do nothing: Can't add any more prefill anyway
            return SchedulerPrefillOutputs(
                seq_groups=[],
                ignored_seq_groups=[],
                num_lookahead_slots=self._get_num_lookahead_slots(
                    is_prefill=True, enable_chunking=enable_chunking),
            )
        ignored_seq_groups: List[SequenceGroup] = []
        seq_groups: List[ScheduledSequenceGroup] = []

        waiting_queue = self.waiting

        leftover_waiting_sequences: Deque[SequenceGroup] = deque()
        while self._passed_delay(time.time()) and waiting_queue:
            seq_group = waiting_queue[0]

            waiting_seqs = seq_group.get_seqs(status=SequenceStatus.WAITING)
            assert len(waiting_seqs) == 1, (
                "Waiting sequence group should have only one prompt "
                "sequence.")
            if (partial_prefill_metadata is not None
                    and not partial_prefill_metadata.can_schedule(seq_group)):
                leftover_waiting_sequences.appendleft(seq_group)
                waiting_queue.popleft()
                continue
            num_new_tokens_uncached, num_new_tokens_cached = (
                self._get_num_new_uncached_and_cached_tokens(
                    seq_group,
                    SequenceStatus.WAITING,
                    enable_chunking,
                    budget,
                    partial_prefill_metadata=partial_prefill_metadata,
                ))
            num_new_tokens = num_new_tokens_uncached + num_new_tokens_cached

            if not enable_chunking:
                num_prompt_tokens = waiting_seqs[0].get_len()
                assert num_new_tokens == num_prompt_tokens

            prompt_limit = self._get_prompt_limit(seq_group)
            if num_new_tokens > prompt_limit:
                logger.warning(
                    "Input prompt (%d tokens) is too long"
                    " and exceeds limit of %d",
                    num_new_tokens,
                    prompt_limit,
                )
                for seq in waiting_seqs:
                    seq.status = SequenceStatus.FINISHED_IGNORED
                ignored_seq_groups.append(seq_group)
                waiting_queue.popleft()
                continue

            num_lookahead_slots: int = 0
            if self.scheduler_config.is_multi_step and enable_chunking:
                num_lookahead_slots = self._get_num_lookahead_slots(
                    True, enable_chunking)

            # If the sequence group cannot be allocated, stop.
            can_allocate = self.block_manager.can_allocate(
                seq_group, num_lookahead_slots=num_lookahead_slots)
            if can_allocate == AllocStatus.LATER:
                break
            elif can_allocate == AllocStatus.NEVER:
                logger.warning(
                    "Input prompt (%d tokens) + lookahead slots (%d) is "
                    "too long and exceeds the capacity of block_manager",
                    num_new_tokens,
                    num_lookahead_slots,
                )
                for seq in waiting_seqs:
                    seq.status = SequenceStatus.FINISHED_IGNORED
                ignored_seq_groups.append(seq_group)
                waiting_queue.popleft()
                continue

            lora_int_id = 0
            if self.lora_enabled:
                lora_int_id = seq_group.lora_int_id
                assert curr_loras is not None
                assert self.lora_config is not None
                if (self.lora_enabled and lora_int_id > 0
                        and lora_int_id not in curr_loras
                        and len(curr_loras) >= self.lora_config.max_loras):
                    # We don't have a space for another LoRA, so
                    # we ignore this request for now.
                    leftover_waiting_sequences.appendleft(seq_group)
                    waiting_queue.popleft()
                    continue

            if (budget.num_batched_tokens
                    >= self.scheduler_config.max_num_batched_tokens):
                # We've reached the budget limit - since there might be
                # continuous prefills in the running queue, we should break
                # to avoid scheduling any new prefills.
                break

            num_new_seqs = seq_group.get_max_num_running_seqs()
            if num_new_tokens_uncached == 0 or not budget.can_schedule(
                    num_new_tokens=num_new_tokens_uncached,
                    num_new_seqs=num_new_seqs,
            ):
                break

            # Can schedule this request.
            if curr_loras is not None and lora_int_id > 0:
                curr_loras.add(lora_int_id)
            waiting_queue.popleft()
            self._allocate_and_set_running(seq_group)

            if partial_prefill_metadata is not None:
                partial_prefill_metadata.maybe_increment_partial_prefills(
                    seq_group)

            if enable_chunking and self.scheduler_config.is_multi_step:
                blocks_to_copy: List[Tuple[int, int]] = []
                # init_multi_step_from_lookahead_slots happens in append_slots
                self._append_slots(seq_group, blocks_to_copy, enable_chunking)
                # This assert will trip when a copy-on-write happens. This is
                # not a concern as the very first sequence-group block
                # allocation happens above. Still, we have the assert to
                # catch any edge-cases.
                assert not blocks_to_copy
            else:
                seq_group.init_multi_step_from_lookahead_slots(
                    num_lookahead_slots,
                    num_scheduler_steps=self.scheduler_config.
                    num_scheduler_steps,
                    is_multi_step=self.scheduler_config.is_multi_step,
                    enable_chunking=enable_chunking,
                )

            seq_groups.append(
                ScheduledSequenceGroup(seq_group=seq_group,
                                       token_chunk_size=num_new_tokens))
            budget.add_num_batched_tokens(
                seq_group.request_id,
                num_batched_tokens=num_new_tokens_uncached,
                num_cached_tokens=num_new_tokens_cached,
            )
            budget.add_num_seqs(seq_group.request_id, num_new_seqs)

            # NOTE(RBLN):
            # For rbln target, we only consider batch size of 1 for prefill.
            break

        logger.debug("waiting_queue -> len=%s", len(waiting_queue))
        # Queue requests that couldn't be scheduled.
        waiting_queue.extendleft(leftover_waiting_sequences)
        if len(seq_groups) > 0:
            self.prev_prompt = True

        return SchedulerPrefillOutputs(
            seq_groups=seq_groups,
            ignored_seq_groups=ignored_seq_groups,
            num_lookahead_slots=self._get_num_lookahead_slots(
                is_prefill=True, enable_chunking=enable_chunking),
        )

    def _schedule_chunked_prefill(self) -> SchedulerOutputs:
        """Schedule queued requests.

        Chunked prefill allows to chunk prefill requests, batch them together
        with decode requests. This policy 1. schedule as many decoding requests
        as possible. 2. schedule chunked prefill requests that are not
        finished. 3. schedule swapped request. 4. schedule new prefill
        requests.

        The policy can sustain the high GPU utilization because it can put
        prefill and decodes requests to the same batch, while it improves
        inter token latency because decodes requests don't need to be blocked
        by prefill requests.
        """
        budget = SchedulingBudget(
            token_budget=self.scheduler_config.max_num_batched_tokens,
            max_num_seqs=self.scheduler_config.max_num_seqs,
        )
        curr_loras: Set[int] = set()

        prefills = SchedulerPrefillOutputs.create_empty()
        running_scheduled = SchedulerRunningOutputs.create_empty()
        swapped_in = SchedulerSwappedInOutputs.create_empty()

        # Create partial prefill metadata
        partial_prefill_metadata = \
            RBLNPartialPrefillMetadata.from_queues(
                running=self.running,
                waiting=self.waiting,
                scheduler_config=self.scheduler_config,
            )

        # NOTE(jiwoo.park): If there are remaining Prefills,
        # perform all of them as a single batch first,
        # then perform decoding as a multi-batch.
        if partial_prefill_metadata.schedulable_prefills == 0 or (
                len(self.running) > 0 and self.running[0].first_seq.data.stage
                == SequenceStage.PREFILL):
            # Decoding should be always scheduled first by fcfs.
            running_scheduled = self._schedule_running(
                budget,
                curr_loras,
                enable_chunking=True,
                partial_prefill_metadata=partial_prefill_metadata,
            )

            if partial_prefill_metadata.schedulable_prefills == 0 and \
                len(running_scheduled.preempted) + len(
                        running_scheduled.swapped_out) == 0:
                swapped_in = self._schedule_swapped(budget, curr_loras)
        else:
            prefills = self._schedule_prefills(
                budget,
                curr_loras,
                enable_chunking=True,
                partial_prefill_metadata=partial_prefill_metadata,
            )

        assert (budget.num_batched_tokens
                <= self.scheduler_config.max_num_batched_tokens)
        assert (
            budget.num_curr_seqs <= self.scheduler_config.max_num_seqs
        ), f"{budget.num_curr_seqs} <= {self.scheduler_config.max_num_seqs}"

        # Update waiting requests.
        self.waiting.extendleft(running_scheduled.preempted)

        # NOTE(jiwoo.park) RBLN device can't schedule
        # decode and prefill requests together.
        # So, our scheduler prioritizes prefills, not decode priority.
        # Update new running requests.
        # By default, vLLM scheduler prioritizes prefills.
        # Once chunked prefill is enabled,
        # the policy is changed to prioritize decode requests.
        # self.running.extend(
        #     [s.seq_group for s in swapped_in.decode_seq_groups])
        # self.running.extend(
        #     [s.seq_group for s in swapped_in.prefill_seq_groups])
        # self.running.extend(
        #     [s.seq_group for s in running_scheduled.decode_seq_groups])
        # Because multiple prefills may be running concurrently, we need to
        # make sure that prefills which are scheduled to finish are listed
        # before those that won't. This is so that on the next scheduling
        # iteration when they have transitioned to the decode stage, they are
        # properly prioritized over sequences that are still in the prefill
        # stage.
        # self.running.extend(
        #     self._order_finishing_prefills_first(
        #         running_scheduled.prefill_seq_groups))
        # self.running.extend([s.seq_group for s in prefills.seq_groups])
        finishing, not_finishing = self._split_prefills(
            running_scheduled.prefill_seq_groups)
        self.running.extendleft([s.seq_group for s in prefills.seq_groups])
        self.running.extendleft(not_finishing)
        self.running.extendleft(
            [s.seq_group for s in swapped_in.prefill_seq_groups])
        self.running.extend(
            [s.seq_group for s in swapped_in.decode_seq_groups])
        self.running.extend(
            [s.seq_group for s in running_scheduled.decode_seq_groups])
        self.running.extend(finishing)

        # Update swapped requests.
        self.swapped.extend(running_scheduled.swapped_out)
        # Put prefills first due to Attention backend ordering assumption.
        scheduled_seq_groups = (prefills.seq_groups +
                                running_scheduled.prefill_seq_groups +
                                swapped_in.prefill_seq_groups +
                                running_scheduled.decode_seq_groups +
                                swapped_in.decode_seq_groups)
        num_prefill_groups = (len(prefills.seq_groups) +
                              len(swapped_in.prefill_seq_groups) +
                              len(running_scheduled.prefill_seq_groups))
        # If all prompts, then we set num_lookahead_slots to 0
        # this allows us to go through the `no_spec` path in
        # `spec_decode_worker.py`
        all_prefills = len(scheduled_seq_groups) == num_prefill_groups
        num_lookahead_slots = (0 if
                               (all_prefills
                                and not self.scheduler_config.is_multi_step)
                               else running_scheduled.num_lookahead_slots)
        return SchedulerOutputs(
            scheduled_seq_groups=scheduled_seq_groups,
            num_prefill_groups=num_prefill_groups,
            num_batched_tokens=budget.num_batched_tokens +
            budget.num_cached_tokens,
            blocks_to_swap_in=swapped_in.blocks_to_swap_in,
            blocks_to_swap_out=running_scheduled.blocks_to_swap_out,
            blocks_to_copy=running_scheduled.blocks_to_copy +
            swapped_in.blocks_to_copy,
            ignored_seq_groups=prefills.ignored_seq_groups +
            swapped_in.infeasible_seq_groups,
            num_lookahead_slots=num_lookahead_slots,
            running_queue_size=len(self.running),
            preempted=(len(running_scheduled.preempted) +
                       len(running_scheduled.swapped_out)),
        )

    def _split_prefills(
        self, scheduled_prefill_seqs: List[ScheduledSequenceGroup]
    ) -> Tuple[List[SequenceGroup], List[SequenceGroup]]:
        finishing = [
            s.seq_group for s in scheduled_prefill_seqs
            if s.seq_group.get_num_uncomputed_tokens() == s.token_chunk_size
        ]
        not_finishing = [
            s.seq_group for s in scheduled_prefill_seqs
            if s.seq_group.get_num_uncomputed_tokens() != s.token_chunk_size
        ]
        return finishing, not_finishing

    def _can_append_slots(self, seq_group: SequenceGroup,
                          enable_chunking: bool) -> bool:
        """Determine whether or not we have enough space in the KV cache to
        continue generation of the sequence group.
        """
        """
        FIXME(RBLN):
        This is code from the legacy vLLM (RBLN).
        It seems unnecessary in the vLLM RBLN plugin,
        but I'm not entirely sure.
        So, I commented it out for now.
        If it is required, the comment should be removed.
        """
        # always append for RBLN eager attention
        # if self.cache_config.block_size == \
        #     self.scheduler_config.max_model_len:
        #     return True

        # It is True only for testing case to trigger artificial preemption.
        if (self.enable_artificial_preemption
                and random.uniform(0, 1) < ARTIFICIAL_PREEMPTION_PROB
                and self.artificial_preempt_cnt > 0):
            self.artificial_preempt_cnt -= 1
            return False

        is_prefill = seq_group.is_prefill()
        num_lookahead_slots = self._get_num_lookahead_slots(
            is_prefill, enable_chunking)

        if is_prefill and num_lookahead_slots > 0:
            # Appending prefill slots only happens multi-step and
            # chunked-prefill are enabled together.
            assert self.scheduler_config.is_multi_step and enable_chunking

        return self.block_manager.can_append_slots(
            seq_group=seq_group, num_lookahead_slots=num_lookahead_slots)


@dataclass
class RBLNPartialPrefillMetadata(PartialPrefillMetadata):

    @classmethod
    def from_queues(
        cls,
        running: Deque[SequenceGroup],
        waiting: Deque[SequenceGroup],
        scheduler_config: SchedulerConfig,
    ) -> "RBLNPartialPrefillMetadata":
        """Create a PartialPrefillMetadata object from the current state of
        the scheduler's queues.
        This accounts for the currently running prefill requests, and peeks into
        the waiting queue to see if there are more prefills to potentially be
        scheduled during this iteration."""
        prefills = 0
        long_prefills = 0

        waiting_long_prefills = 0

        decodings = 0

        for sg in running:
            if sg.first_seq.data.stage == SequenceStage.PREFILL:
                prefills += 1
                if sg.first_seq.get_num_new_tokens(
                ) > scheduler_config.long_prefill_token_threshold:
                    long_prefills += 1
            # NOTE(jiwoo.park):
            # count running decode requests for the decoding-priority policy
            else:
                decodings += 1

        # NOTE(jiwoo.park): decoding-priority policy
        # Introduce a policy where prefill is prioritized,
        # but if the number of pending decodings
        # reaches the maximum number of sequences (`max_num_seqs`)
        # that can be processed, decoding is performed first.
        if decodings >= scheduler_config.max_num_seqs:
            return RBLNPartialPrefillMetadata(
                schedulable_prefills=0,
                long_prefills=0,
                scheduler_config=scheduler_config)

        for sg in waiting:
            # Don't bother looping through the rest of the queue if we know
            # there are already at
            # least max_partial_prefills requests to fill
            if prefills >= scheduler_config.max_num_partial_prefills:
                break

            # Don't count long requests from the waiting queue if we aren't
            # going to schedule them anyway
            if sg.first_seq.get_num_new_tokens(
            ) > scheduler_config.long_prefill_token_threshold:
                if long_prefills + waiting_long_prefills >= \
                    scheduler_config.max_long_partial_prefills:
                    continue
                waiting_long_prefills += 1
            prefills += 1

        # NB: long_prefills and waiting_long_prefills are tracked separately.
        # We don't account for the waiting requests here because we need to use
        # this metadata to track how many have actually been scheduled.
        return RBLNPartialPrefillMetadata(
            schedulable_prefills=min(
                prefills, scheduler_config.max_num_partial_prefills),
            long_prefills=long_prefills,
            scheduler_config=scheduler_config,
        )
