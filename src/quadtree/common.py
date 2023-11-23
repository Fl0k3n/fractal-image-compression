from dataclasses import dataclass, replace

NORMAL_ORIENTATION = 0
TRANSPOSED_ORIENTATION = 1
MAX_GRAY = 255

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
class EncodingInfo:
    scale_bits: int
    offset_bits: int
    max_scale: float
    img_width: int
    img_height: int

    def copy(self) -> "EncodingInfo":
        return replace(self)

class QuadtreeNode:
    def __init__(self, domain: Domain | None, children: list["QuadtreeNode", "QuadtreeNode", "QuadtreeNode", "QuadtreeNode"] | None) -> None:
        self.domain = domain
        self.children = children
    
    def is_leaf(self):
        return self.domain is not None
    
    @staticmethod
    def Leaf(domain: Domain) -> "QuadtreeNode":
        return QuadtreeNode(domain, None)
    
    @staticmethod
    def ParentOf(c1: "QuadtreeNode", c2: "QuadtreeNode", c3: "QuadtreeNode", c4: "QuadtreeNode") -> "QuadtreeNode":
        return QuadtreeNode(None, [c1, c2, c3, c4])

@dataclass
class QuadtreeImage:
    info: EncodingInfo
    forest: list[QuadtreeNode]
