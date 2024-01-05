from dataclasses import dataclass
from typing import Optional

from quadtree.common import EncodingInfo


@dataclass
class Domain:
    start_i: int
    start_j: int
    orientation: int
    rotation: int
    scale: float # for testing/debugging only
    offset: float
    quantized_scale: int
    quantized_offset: int

@dataclass
class SplitInfo:
    vertical_split: bool
    split_idx: int

class HVNode:
    def __init__(self, c1: Optional["HVNode"], c2: Optional["HVNode"],
                split_info: Optional[SplitInfo], domain: Optional[Domain]) -> None:
        self.c1 = c1
        self.c2 = c2
        self.split_info = split_info
        self.domain = domain

    def is_leaf(self) -> bool:
        return self.domain is not None
    
    @staticmethod
    def ParentOf(c1: "HVNode", c2: "HVNode", split_info: SplitInfo) -> "HVNode":
        return HVNode(c1, c2, split_info, None)
    
    @staticmethod
    def Leaf(domain: Domain) -> "HVNode":
        return HVNode(None, None, None, domain)
    
@dataclass
class HVImage:
    info: EncodingInfo
    root: HVNode
