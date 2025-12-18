from .optimum_block_configuration import BlockConfiguration
from .optimum_block_mapping_manager import BlockMappingManager
from dataclasses import dataclass
from vllm_rbln.logger import init_logger
import logging
import time
logger = init_logger(__name__)


@dataclass
class CacheSearchResult:
    cached_outer_blocks: list[int]
    cached_lengths: list[int]

    @property
    def has_cache_hit(self) -> bool:
        return len(self.cached_outer_blocks) > 0

class InnerBlockGroupManager:
    """
    The blocks with same block hashes are grouped together.
    """
    def __init__(self):
        self.groups: dict[str, set[int]] = {}
        # group_id -> outer block ids
        self.outer_block_ids: dict[str, list[int]] = {}
        # group_id -> outer block offset
        self.outer_block_offsets: dict[str, int] = {}
        self.inner_to_group_id: dict[int, int] = {}

    def register_inner_block(self, inner_block_id: int, same_inner_block_id: int, outer_block_id: int, outer_block_offset: int) -> None:
        assert inner_block_id not in self.inner_to_group_id, \
            f"IB: {inner_block_id} already registered"
        if same_inner_block_id is None:
            group_id = self.get_new_group_id()
            self.outer_block_ids[group_id] = [outer_block_id]
            self.outer_block_offsets[group_id] = outer_block_offset
        else:
            group_id = self.get_group_id(same_inner_block_id)
            assert inner_block_id not in self.groups[group_id], \
                f"IB: {inner_block_id} already in group {group_id}"
            assert outer_block_offset == self.outer_block_offsets[group_id], \
                f"OB: {outer_block_offset} != {self.outer_block_offsets[group_id]} in group {group_id}"
            if outer_block_id not in self.outer_block_ids[group_id]:
                self.outer_block_ids[group_id].append(outer_block_id)
        self.set_group_id(inner_block_id, group_id)

    def unregister_inner_block(self, inner_block_id: int) -> None:
        group_id = self.get_group_id(inner_block_id)
        self.groups[group_id].remove(inner_block_id)
        self.inner_to_group_id.pop(inner_block_id)
        if len(self.groups[group_id]) == 0:
            self.groups.pop(group_id)

    def set_group_id(self, inner_block_id: int, group_id: int) -> None:
        self.inner_to_group_id[inner_block_id] = group_id
        self.groups[group_id].add(inner_block_id)

    def get_group_id(self, inner_block_id: int) -> int:
        return self.inner_to_group_id.get(inner_block_id)
    
    def get_new_group_id(self) -> int:
        group_id = str(time.time())
        self.groups[group_id] = set[int]()
        return group_id

    def get_group_ids(self, inner_block_ids: list[int]) -> list[int]:
        return [self.get_group_id(ib_id) for ib_id in inner_block_ids]


class CacheHistoryManager:
    """
    Search for cached blocks that can be reused.
    It maintains a per-request cache of search results
    until the request is freed.
    """

    def __init__(self, config: BlockConfiguration):
        self._config = config
        self._inner_block_group_manager = InnerBlockGroupManager()

    def register_cached_blocks(self, new_inner_blocks: list[int], cached_blocks: list[int], allocated_outer_blocks: list[int], num_new_computed_tokens: int) -> None:
        if len(cached_blocks) == 0:
            for i, new_inner_block in enumerate(new_inner_blocks):
                ob_id = allocated_outer_blocks[i // self._config.block_ratio]
                ob_offset = i % self._config.block_ratio
                self._inner_block_group_manager.register_inner_block(new_inner_block, None, ob_id, ob_offset)
        else:
            # FIXME
            recorded_new_inner_blocks = new_inner_blocks[:len(cached_blocks)]
            # assert len(new_inner_blocks) == len(cached_blocks), \
            #     f"New inner blocks ({len(new_inner_blocks)}) and cached blocks ({len(cached_blocks)}) must have the same length"
            for i, (new_inner_block, cached_block) in enumerate(zip(recorded_new_inner_blocks, cached_blocks)):
                ob_id = allocated_outer_blocks[i // self._config.block_ratio]
                # NOTE: It works because this function is called only in prefill phase.
                ob_offset = i % self._config.block_ratio

                self._inner_block_group_manager.register_inner_block(new_inner_block, cached_block, ob_id, ob_offset)
        

    def unregister_inner_blocks(self, inner_blocks: list[int]) -> None:
        for inner_block in inner_blocks:
            self._inner_block_group_manager.unregister_inner_block(inner_block)

    def find_cached_blocks(
            self, request_id: str, cached_blocks: list[int],
            skip_blocks: set[int], num_new_computed_tokens: int,
            mapping_manager: BlockMappingManager) -> CacheSearchResult:
        """
        Find cached outer blocks that match the given inner blocks.
        """

        best_match = self._try_match_request(request_id, cached_blocks, skip_blocks,
                                             mapping_manager)
        # self._cached_blocks_per_request[request_id] = best_match
        final_num_cached_tokens = sum(best_match.cached_lengths)

        if logger.isEnabledFor(logging.DEBUG):
            if final_num_cached_tokens < num_new_computed_tokens:
                miss_rate = (
                    (num_new_computed_tokens - final_num_cached_tokens) /
                    num_new_computed_tokens *
                    100) if num_new_computed_tokens > 0 else 0
                logger.debug(
                    "[PFX] [CACHE-PARTIAL] REQUEST=%s | "
                    "REUSED=%d/%d tokens (%.1f%%) | "
                    "MISSED=%d tokens (%.1f%%) | "
                    "REASON=partial_cache_miss", request_id,
                    final_num_cached_tokens, num_new_computed_tokens,
                    (final_num_cached_tokens / num_new_computed_tokens * 100),
                    num_new_computed_tokens - final_num_cached_tokens,
                    miss_rate)

            if best_match.has_cache_hit:
                cached_blocks = []
                ob_to_ib_mapping = {}

                for i, (ob, cached_length) in \
                    enumerate(zip(best_match.cached_outer_blocks,
                        best_match.cached_lengths)):
                    ibs = mapping_manager.get_cached_inner_blocks_for_outer(ob)
                    cached_num_ib = cached_length // self._config.ib_size
                    cached_inner_blocks = ibs[:cached_num_ib]
                    cached_blocks.extend(cached_inner_blocks)
                    ob_to_ib_mapping[ob] = len(cached_inner_blocks)

                # TODO specify the hit ratio?
                logger.debug(
                    "[PFX] [CACHE-HIT] REQUEST=%s | "
                    "OB_COUNT=%d OB=%s | "
                    "IB_COUNT=%d IB=%s | "
                    "OB_TO_IB_MAP=%s", request_id,
                    len(best_match.cached_outer_blocks),
                    best_match.cached_outer_blocks, len(cached_blocks),
                    cached_blocks, ob_to_ib_mapping)

        return best_match

    def _try_match_request(
            self, request_id: str, cached_ib: list[int], skip_blocks: set[int],
            mapping_manager: BlockMappingManager) -> CacheSearchResult:
        """
        Try to find the best matching outer blocks for the given inner blocks.
        NOTE Currently, we only support exact match of the inner blocks
        """
        cur_group_ids = self._inner_block_group_manager.get_group_ids(cached_ib)
        final_request_id = None
        max_cache_hit_num_ibs = 0
        max_cache_hit_obs = []
        for other_request_id in mapping_manager.get_request_ids():
            if other_request_id == request_id:
                continue
            outer_blocks = mapping_manager.get_request_blocks(other_request_id)
            inner_blocks = mapping_manager.get_inner_blocks_for_request(other_request_id)
            other_group_ids = self._inner_block_group_manager.get_group_ids(inner_blocks)
            common_prefix = self._get_common_prefix(cur_group_ids, other_group_ids)

            cache_hit_num_ibs = len(common_prefix)
            if cache_hit_num_ibs > max_cache_hit_num_ibs:
                max_cache_hit_num_ibs = cache_hit_num_ibs
                cache_hit_num_obs = cache_hit_num_ibs // self._config.block_ratio
                max_cache_hit_obs = outer_blocks[:cache_hit_num_obs]
        
        cached_lengths = []
        if max_cache_hit_num_ibs > 0:
            last_cache_hit_num_ibs = max_cache_hit_num_ibs % self._config.block_ratio
            cached_lengths = [self._config.ib_size] * (len(max_cache_hit_num_ibs) - 1) + [last_cache_hit_num_ibs * self._config.ib_size]

        return CacheSearchResult(cached_outer_blocks=max_cache_hit_obs,
                                 cached_lengths=cached_lengths)

    def _get_common_prefix(self, arr1: list[int],
                           arr2: list[int]) -> list[int]:
        """
        Return the common prefix between two lists of integers.
        """
        common_prefix = []
        min_length = min(len(arr1), len(arr2))
        for i in range(min_length):
            if arr1[i] == arr2[i]:
                common_prefix.append(arr1[i])
            else:
                break
        return common_prefix