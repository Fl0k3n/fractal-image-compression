from dataclasses import dataclass, replace


@dataclass
class HvDomain:
    start_i: int
    start_j: int

@dataclass
class HvPartition:
    domain_idx: int  # which domain it references
    type: int  #  0 - horizontal, 1 - vertical
    position: int #  num of pixels from edge of the partition
    orientation: int
    rotation: int
    s: float
    o: float

@dataclass
class HvRange:
    start_i: int
    start_j: int
    len_i: int
    len_j: int
    domain_idx: int
    orientation: int
    rotation: int
    s: float
    o: float

@dataclass
class EncodedHvImage:
    def __init__(self, width: int, height: int) -> None:
        self.domains: list[HvDomain] = []
        self.width: int = width
        self.height: int = height
        self.smallest_partitioned: int = 0
        self.partitions: list[HvPartition] = []
        self.ranges: list[HvRange] = []
        
