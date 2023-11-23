import struct
from collections import deque
from typing import BinaryIO

import numpy as np

from bitbuffer import EOF, BitBuffer
from quadtree.common import Domain, EncodingInfo, QuadtreeImage, QuadtreeNode

QUADTREE_MAGIC = "quadtree".encode(encoding='ascii')
INTERNAL_NODE_BIT = 0
LEAF_BIT = 1

STRUCT_BYTE_ORDER = "!"
HEADER_STRUCT_FMT = f'{STRUCT_BYTE_ORDER}{len(QUADTREE_MAGIC)}sfBBHH'
HEADER_SIZE = len(QUADTREE_MAGIC) + 10
UNSUPPORTED = 0.

class QuadtreeSerializer:
    def __init__(self, quadtree_img: QuadtreeImage) -> None:
        self.img = quadtree_img
        self.file: BinaryIO = None 
        
        self.domain_i_bits = int(np.ceil(np.log2(self.img.info.img_height)))
        self.domain_j_bits = int(np.ceil(np.log2(self.img.info.img_width)))

    def serialize(self, output: str | BinaryIO):
        self.file = output
        should_close = False
        try:
            if isinstance(output, str):
                should_close = True
                self.file = open(output, 'wb')
            self._write_header() 
            self._write_data() 
        finally:
            if should_close:
                try:
                    self.file.close()
                except:
                    pass
    
    def _write_header(self):
        header = struct.pack(HEADER_STRUCT_FMT, QUADTREE_MAGIC, self.img.info.max_scale,
                    self.img.info.scale_bits, self.img.info.offset_bits, 
                    self.img.info.img_width, self.img.info.img_height)
        self.file.write(header)

    def _write_data(self):
        buff = BitBuffer(self.file, STRUCT_BYTE_ORDER)
        for root in self.img.forest:
            queue = deque[QuadtreeNode]()
            queue.append(root)
            while queue:
                cur = queue.popleft()
                if cur.is_leaf():
                    dom = cur.domain
                    buff.write(LEAF_BIT, 1)
                    buff.write(dom.start_i, self.domain_i_bits)
                    buff.write(dom.start_j, self.domain_j_bits)
                    buff.write(dom.orientation, 1)
                    buff.write(dom.rotation, 2)
                    buff.write(dom.quantized_scale, self.img.info.scale_bits)
                    buff.write(dom.quantized_offset, self.img.info.offset_bits)
                else:
                    buff.write(INTERNAL_NODE_BIT, 1)
                    for c in cur.children:
                        queue.append(c)
        if not buff.is_empty():
            buff.flush()

class QuadtreeDeserializer:
    def __init__(self) -> None:
        self.file: BinaryIO = None
        self.img: QuadtreeImage = None
        self.domain_i_bits = -1
        self.domain_j_bits = -1

    def deserialize(self, input: str | BinaryIO) -> QuadtreeImage:
        self.file = input
        should_close = False
        self.img = QuadtreeImage(None, None)
        try:
            if isinstance(input, str):
                should_close = True
                self.file = open(input, 'rb')
            self._read_header() 
            self._read_data() 
            return self.img
        finally:
            if should_close:
                try:
                    self.file.close()
                except:
                    pass
    
    def _read_header(self):
        serialized_header = self.file.read(HEADER_SIZE)
        if len(serialized_header) != HEADER_SIZE:
            raise Exception(f'Invalid file format: header is too short, empty file?')

        header = struct.unpack(HEADER_STRUCT_FMT, serialized_header)
        if header[0] != QUADTREE_MAGIC:
            raise Exception(f'Invalid file format: file is missing quadtree magic')

        self.img.info = EncodingInfo(
            scale_bits=header[2],
            offset_bits=header[3],
            max_scale=header[1],
            img_width=header[4],
            img_height=header[5]
        )
        self.domain_i_bits = int(np.ceil(np.log2(self.img.info.img_height)))
        self.domain_j_bits = int(np.ceil(np.log2(self.img.info.img_width)))
        
    def _read_data(self):
        self.img.forest = []
        buff = BitBuffer(self.file, STRUCT_BYTE_ORDER)
        cur_root = None
        parent_queue = deque[tuple[QuadtreeNode, int]]()

        while (node_type := buff.read(1)) != EOF:
            if node_type == LEAF_BIT:
                start_i = buff.read(self.domain_i_bits)
                start_j = buff.read(self.domain_j_bits)
                orientation = buff.read(1)
                rotation = buff.read(2)
                quantized_scale = buff.read(self.img.info.scale_bits)
                quantized_offset = buff.read(self.img.info.offset_bits)
                node = QuadtreeNode.Leaf(Domain(
                    start_i, start_j, orientation, rotation, scale=UNSUPPORTED, offset=UNSUPPORTED,
                    quantized_scale=quantized_scale, quantized_offset=quantized_offset
                ))
                if parent_queue:
                    parent, child_idx = parent_queue.popleft()
                    parent.children[child_idx] = node
                else:
                    if cur_root is not None:
                        self.img.forest.append(cur_root)
                        cur_root = None
                    self.img.forest.append(node)
            else:
                node = QuadtreeNode.ParentOf(None, None, None, None)
                if parent_queue:
                    parent, child_idx = parent_queue.popleft()
                    parent.children[child_idx] = node
                else:
                    if cur_root is not None:
                        self.img.forest.append(cur_root)
                    cur_root = node
                for i in range(4):
                    parent_queue.append((node, i))

        if cur_root is not None and not parent_queue:
            self.img.forest.append(cur_root)    
