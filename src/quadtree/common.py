from dataclasses import dataclass, replace
from typing import Callable

import numpy as np

from entropy.processor import CodingDimension

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

@dataclass
class ColoredQuadtreeImage:
    encoded_Y: QuadtreeImage
    encoded_Cb: QuadtreeImage
    encoded_Cr: QuadtreeImage

def get_dimension_bit_widths(info: EncodingInfo) -> list[int]:
    dimension_default_bit_widths = [None for _ in CodingDimension]
    dimension_default_bit_widths[CodingDimension.DOMAIN_ROW.value] = int(np.ceil(np.log2(info.img_height)))
    dimension_default_bit_widths[CodingDimension.DOMAIN_COL.value] = int(np.ceil(np.log2(info.img_width)))
    dimension_default_bit_widths[CodingDimension.SCALE.value] = info.scale_bits
    dimension_default_bit_widths[CodingDimension.OFFSET.value] = info.offset_bits
    return dimension_default_bit_widths

def get_domain_dimension_extractors() -> list[Callable[[Domain], int]]:
    dimension_extractors = [None for _ in CodingDimension]
    dimension_extractors[CodingDimension.DOMAIN_ROW.value] = lambda d: d.start_i
    dimension_extractors[CodingDimension.DOMAIN_COL.value] = lambda d: d.start_j
    dimension_extractors[CodingDimension.SCALE.value] =      lambda d: d.quantized_scale
    dimension_extractors[CodingDimension.OFFSET.value] =     lambda d: d.quantized_offset
    return dimension_extractors
