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
"""CPU affinity utilities for RBLN worker."""

import os
import platform
from collections.abc import Callable

from vllm.config import ParallelConfig
from vllm.platforms import CpuArchEnum, current_platform
from vllm.platforms.cpu import CpuPlatform, LogicalCPUInfo

import vllm_rbln.rbln_envs as envs
from vllm_rbln.logger import init_logger

logger = init_logger(__name__)


def get_autobind_cpu_ids(
    rank: int,
    local_rank: int,
    parallel_config: ParallelConfig,
    cpu_selector: Callable[[list[LogicalCPUInfo]], list[LogicalCPUInfo]],
) -> str:
    """Get CPU IDs for automatic thread binding based on NUMA nodes.

    Args:
        rank: Global rank of the worker.
        local_rank: Local rank of the worker.
        parallel_config: Parallel configuration.
        cpu_selector: Function to select CPUs from each physical core.

    Returns:
        Comma-separated string of CPU IDs, or "all" or "nobind".
    """
    allowed_numa_nodes, logical_cpu_list = (
        CpuPlatform.get_allowed_cpu_core_node_list())

    # Calculate rank_across_dp for CPU binding
    # This ensures different DP groups get different CPU allocations
    world_size = parallel_config.world_size
    if parallel_config.data_parallel_size > 1:
        world_size_across_dp = parallel_config.world_size_across_dp
        dp_rank = parallel_config.data_parallel_rank
        rank_across_dp = dp_rank * world_size + local_rank
    else:
        world_size_across_dp = world_size
        rank_across_dp = rank

    # Group CPUs by NUMA node
    numa_node_to_cpus: dict[int, list[LogicalCPUInfo]] = {}
    for cpu_info in logical_cpu_list:
        numa_node = cpu_info.numa_node
        if numa_node not in numa_node_to_cpus:
            numa_node_to_cpus[numa_node] = []
        numa_node_to_cpus[numa_node].append(cpu_info)

    # Filter to only allowed NUMA nodes
    available_numa_nodes = [
        n for n in allowed_numa_nodes if n in numa_node_to_cpus
    ]

    if not available_numa_nodes:
        logger.error(
            "Auto thread-binding failed: no available NUMA nodes "
            "with allowed CPUs. Please try to bind threads manually.")
        return "all"

    numa_node_idx = rank_across_dp % len(available_numa_nodes)
    selected_numa_node = available_numa_nodes[numa_node_idx]
    numa_node_cpu_list = numa_node_to_cpus[selected_numa_node]
    ranks_in_same_numa = [
        r for r in range(world_size_across_dp)
        if r % len(available_numa_nodes) == numa_node_idx
    ]

    # Select CPUs from each physical core via cpu_selector
    core_to_cpus: dict[int, list[LogicalCPUInfo]] = {}
    for cpu_info in numa_node_cpu_list:
        if cpu_info.physical_core not in core_to_cpus:
            core_to_cpus[cpu_info.physical_core] = []
        core_to_cpus[cpu_info.physical_core].append(cpu_info)
    selected_cpu_list = []
    for cpu_list in core_to_cpus.values():
        cpu_list = sorted(cpu_list, key=lambda x: x.id)
        selected_cpu_list.extend(cpu_selector(cpu_list))
    selected_cpu_list = sorted(selected_cpu_list, key=lambda x: x.id)

    # Always divide CPUs among ranks in the same NUMA node
    # for exclusive allocation
    if len(ranks_in_same_numa) > 1:
        cpus_per_rank = len(selected_cpu_list) // len(ranks_in_same_numa)
        remainder = len(selected_cpu_list) % len(ranks_in_same_numa)

        rank_position = ranks_in_same_numa.index(rank_across_dp)
        start_idx = rank_position * cpus_per_rank + min(
            rank_position, remainder)
        end_idx = (start_idx + cpus_per_rank +
                   (1 if rank_position < remainder else 0))
        logical_cpu_list = selected_cpu_list[start_idx:end_idx]
    else:
        logical_cpu_list = selected_cpu_list

    if not logical_cpu_list:
        logger.warning(
            "Auto thread-binding: no CPUs allocated for rank %d "
            "(rank_across_dp %d). Falling back to default.",
            rank,
            rank_across_dp,
        )
        return "all"

    # Log binding information
    if len(ranks_in_same_numa) > 1:
        logger.info(
            "auto thread-binding: rank %d (rank_across_dp %d) "
            "-> NUMA node %d, CPUs: %s (exclusive allocation, "
            "shared NUMA node with ranks %s, id, physical core): %s",
            rank,
            rank_across_dp,
            selected_numa_node,
            ",".join(str(x.id) for x in logical_cpu_list),
            [r for r in ranks_in_same_numa if r != rank_across_dp],
            [(x.id, x.physical_core) for x in logical_cpu_list],
        )
    else:
        logger.info(
            "auto thread-binding: rank %d (rank_across_dp %d) "
            "-> NUMA node %d, CPUs: %s (exclusive allocation, "
            "id, physical core): %s",
            rank,
            rank_across_dp,
            selected_numa_node,
            ",".join(str(x.id) for x in logical_cpu_list),
            [(x.id, x.physical_core) for x in logical_cpu_list],
        )

    return ",".join([str(x.id) for x in logical_cpu_list])


def set_cpu_affinity(
    rank: int,
    local_rank: int,
    parallel_config: ParallelConfig,
) -> None:
    """Setup thread affinity based on NUMA nodes.

    Args:
        rank: Global rank of the worker.
        local_rank: Local rank of the worker.
        parallel_config: Parallel configuration.
    """
    # Setup thread affinity based on NUMA nodes
    if envs.VLLM_RBLN_NUMA and platform.system() == "Linux":
        cpu_arch = current_platform.get_cpu_architecture()
        if cpu_arch in (CpuArchEnum.POWERPC, CpuArchEnum.S390X):
            # For S390X/POWERPC SMT-8/4/2
            local_omp_cpuid = get_autobind_cpu_ids(
                rank,
                local_rank,
                parallel_config,
                lambda cpus: [cpu for cpu in cpus if cpu.id % 8 < 4])
        elif cpu_arch == CpuArchEnum.X86:
            # For x86 SMT-2, use 1 CPU per core
            local_omp_cpuid = get_autobind_cpu_ids(
                rank,
                local_rank,
                parallel_config,
                lambda cpus: cpus[-1:])
        else:
            local_omp_cpuid = "nobind"
    else:
        local_omp_cpuid = "nobind"

    if local_omp_cpuid not in ("all", "nobind"):
        # Parse CPU IDs from string (e.g., "0,1,2,3" -> [0, 1, 2, 3])
        cpu_ids = [
            int(cpu_id.strip())
            for cpu_id in local_omp_cpuid.split(",")
        ]
        # Set CPU affinity for current process
        try:
            os.sched_setaffinity(0, cpu_ids)
            # Verify CPU affinity was set correctly
            actual_cpu_ids = sorted(os.sched_getaffinity(0))
            expected_cpu_ids = sorted(cpu_ids)
            if actual_cpu_ids != expected_cpu_ids:
                logger.warning(
                    "CPU affinity mismatch for rank %d (local_rank %d): "
                    "expected %s, but got %s",
                    rank,
                    local_rank,
                    expected_cpu_ids,
                    actual_cpu_ids,
                )
            else:
                logger.info(
                    "Set CPU affinity for rank %d (local_rank %d): CPUs %s",
                    rank,
                    local_rank,
                    local_omp_cpuid,
                )
        except OSError as e:
            logger.error(
                "Failed to set CPU affinity for rank %d (local_rank %d): "
                "%s",
                rank,
                local_rank,
                str(e),
            )
            raise
    elif local_omp_cpuid == "nobind":
        logger.info(
            "Skipping CPU affinity binding for rank %d (local_rank %d): "
            "nobind",
            rank,
            local_rank,
        )


def set_omp_num_threads(
    rank: int,
    local_rank: int,
    default_num_threads: int = 2,
) -> None:
    """Set OMP_NUM_THREADS environment variable if not already defined.

    Args:
        rank: Global rank of the worker.
        local_rank: Local rank of the worker.
        default_num_threads: Default number of threads to use if
            OMP_NUM_THREADS is not set. Defaults to 2.
    """

    if "OMP_NUM_THREADS" not in os.environ:
        os.environ["OMP_NUM_THREADS"] = str(default_num_threads)
        logger.info(
            "Set OMP_NUM_THREADS to %d for rank %d (local_rank %d)",
            default_num_threads,
            rank,
            local_rank,
        )
    else:
        logger.info(
            "OMP_NUM_THREADS is already defined for rank %d "
            "(local_rank %d): %s",
            rank,
            local_rank,
            os.environ["OMP_NUM_THREADS"],
        )

