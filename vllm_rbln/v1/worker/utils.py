import bisect
from functools import cache

@cache
def select_bucket_size(original_batch_size: int, bucket_sizes: tuple):
    index = bisect.bisect_left(bucket_sizes, original_batch_size)
    return bucket_sizes[index]
