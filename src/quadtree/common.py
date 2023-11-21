from dataclasses import dataclass


@dataclass
class Domain:
    start_i: int
    start_j: int
    orientation: int
    rotation: int
    scale: float
    offset: float

@dataclass
class EncodingInfo:
    scale_bits: int
    offset_bits: int
    max_scale: float

class QuadtreeNode:
    def __init__(self, domain: Domain | None, children: tuple["QuadtreeNode", "QuadtreeNode", "QuadtreeNode", "QuadtreeNode"] | None) -> None:
        self.domain = domain
        self.children = children
    
    def is_leaf(self):
        return self.domain is not None
    
    @staticmethod
    def Leaf(domain: Domain) -> "QuadtreeNode":
        return QuadtreeNode(domain, None)
    
    @staticmethod
    def ParentOf(c1: "QuadtreeNode", c2: "QuadtreeNode", c3: "QuadtreeNode", c4: "QuadtreeNode") -> "QuadtreeNode":
        return QuadtreeNode(None, (c1, c2, c3, c4))
