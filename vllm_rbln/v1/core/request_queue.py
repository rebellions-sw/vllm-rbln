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

from enum import Enum

from vllm.v1.core.sched.request_queue import RequestQueue
from vllm.v1.request import Request
from vllm_rbln.logger import init_logger
from typing import Iterable, Iterator
import heapq

logger = init_logger(__name__)

class RBLNSchedulingPolicy(Enum):
    """Enum for scheduling policies."""
    LJF = "ljf"
    FCFS = "fcfs"
    PRIORITY = "priority"

class LJFRunningRequestQueue(RequestQueue):
    """
    A priority queue that supports heap operations.

    1. requests are ordered by (length, priority, arrival_time).
    """

    def __init__(self) -> None:
        logger.debug("LJFRunningRequestQueue init")
        self._heap: list[tuple[int, int, float, Request]] = []

    def add_request(self, request: Request) -> None:
        """Add a request to the queue according to priority policy."""
        print("_length_key", self._length_key(request))
        key = (self._length_key(request) * -1, request.priority, request.arrival_time)

        heapq.heappush(self._heap, (key, request))

    def pop_request(self) -> Request:
        """Pop a request from the queue according to priority policy."""
        if not self._heap:
            raise IndexError("pop from empty heap")
        return heapq.heappop(self._heap)[1]

    def peek_request(self) -> Request:
        """Peek at the next request in the queue without removing it."""
        if not self._heap:
            raise IndexError("peek from empty heap")
        return self._heap[0][1]

    def prepend_request(self, request: Request) -> None:
        """Add a request to the queue according to priority policy.

        Note: In a priority queue, there is no concept of prepending to the
        front. Requests are ordered by (length, priority, arrival_time)."""
        self.add_request(request)

    def prepend_requests(self, requests: RequestQueue) -> None:
        """Add all requests from another queue according to priority policy.

        Note: In a priority queue, there is no concept of prepending to the
        front. Requests are ordered by (length, priority, arrival_time)."""
        for request in requests:
            self.add_request(request)

    def remove_request(self, request: Request) -> None:
        """Remove a specific request from the queue."""
        self._heap.remove(request)
        heapq.heapify(self._heap)

    def remove_requests(self, requests: Iterable[Request]) -> None:
        """Remove multiple specific requests from the queue."""
        requests_to_remove = requests if isinstance(requests, set) else set(requests)
        self._heap = [r for r in self._heap if r not in requests_to_remove]
        heapq.heapify(self._heap)

    def __bool__(self) -> bool:
        """Check if queue has any requests."""
        return bool(self._heap)

    def __len__(self) -> int:
        """Get number of requests in queue."""
        return len(self._heap)

    def __iter__(self) -> Iterator[Request]:
        """Iterate over the queue according to priority policy."""
        heap_copy = self._heap[:]
        while heap_copy:
            yield heapq.heappop(heap_copy)

    def __reversed__(self) -> Iterator[Request]:
        """Iterate over the queue in reverse priority order."""
        return reversed(list(self))

    def _length_key(self, request: Request) -> int:
        """Get the length of the request."""
        return len(request.all_token_ids)



def create_rbln_request_queue(policy: RBLNSchedulingPolicy) -> RequestQueue:
    """Create a request queue based on the policy."""
    if policy == RBLNSchedulingPolicy.LJF:
        return LJFRunningRequestQueue()
    elif policy == RBLNSchedulingPolicy.PRIORITY:
        return PriorityRequestQueue()
    elif policy == RBLNSchedulingPolicy.FCFS:
        return FCFSRequestQueue()
    else:
        raise ValueError(f"Unknown scheduling policy: {policy}")