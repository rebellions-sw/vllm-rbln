from vllm.v1.core.kv_cache_manager import KVCacheManager


class RBLNOptimumKVCacheManager(KVCacheManager):
    # _allocate_sequence에서는 중간에 BlockTable -> RBLNOptimumBlockTable로 바뀜
    # can_allocate에서는 중간에 if seq_group.is_encoder_decoder():만 빠짐
    # Allocate에서는 if seq_group.is_encoder_decoder():에서 block_table = self._allocate_sequence(encoder_seq) 빠짐
    pass

    # 호출하는 함수
    # get_block_ids
    # get_common_prefix_blocks
    # take_events
    # allocate_slots
    # free
    # get_computed_blocks
    # create_empty_block_list