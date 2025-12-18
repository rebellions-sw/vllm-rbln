from dataclasses import dataclass

@dataclass
class BlockConfiguration:
    ob_size: int
    ib_size: int
    max_model_len: int
    max_num_seqs: int
    num_ob: int

    def __post_init__(self):
        assert self.ob_size % self.ib_size == 0, \
            "ob_size must be a multiple of ib_size"

    @property
    def block_ratio(self) -> int:
        return self.ob_size // self.ib_size
