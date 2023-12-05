import struct
from collections import deque
from typing import BinaryIO

from bitbuffer import EOF, BitBuffer
from entropy.processor import (ENTROPY_PROCESSOR_ID_BITS, CodingDimension,
                               EntropyProcessor, EntropyProcessorId,
                               entropy_processor_factory)
from quadtree.common import (Domain, EncodingInfo, QuadtreeImage, QuadtreeNode,
                             get_dimension_bit_widths,
                             get_domain_dimension_extractors)

QUADTREE_MAGIC = "quadtree".encode(encoding='ascii')
INTERNAL_NODE_BIT = 0
LEAF_BIT = 1

STRUCT_BYTE_ORDER = "!"
HEADER_STRUCT_FMT = f'{STRUCT_BYTE_ORDER}{len(QUADTREE_MAGIC)}sfBBHH'
HEADER_SIZE = len(QUADTREE_MAGIC) + 10
UNSUPPORTED = 0.

DIMIENSION_PRESENT = 1
DIMENSION_MISSING = 0

class QuadtreeSerializer:
    def __init__(self, quadtree_img: QuadtreeImage, entropy_processor: EntropyProcessor,
                 entropy_coding_dimensions: set[CodingDimension]) -> None:
        self.img = quadtree_img
        self.entropy_processor = entropy_processor
        self.entropy_coding_dimensions = entropy_coding_dimensions
        self.file: BinaryIO = None 
        self._init_entropy_processor()

    def serialize(self, output: str | BinaryIO):
        self.file = output
        should_close = False
        try:
            if isinstance(output, str):
                should_close = True
                self.file = open(output, 'wb')
            buff = BitBuffer(self.file, STRUCT_BYTE_ORDER)
            self._write_header(buff) 
            self._write_data(buff) 
        finally:
            if should_close:
                try:
                    self.file.close()
                except:
                    pass

    def _init_entropy_processor(self):
        encoding_dimension_values = self._extract_all_encoding_dimension_values()
        dimension_default_bit_widths = get_dimension_bit_widths(self.img.info)
        self.entropy_processor.prepare_for_encoding(encoding_dimension_values, dimension_default_bit_widths)

    def _extract_all_encoding_dimension_values(self) -> dict[CodingDimension, list[int]]:
        if not self.entropy_coding_dimensions:
            return {}
        queue = deque[QuadtreeNode](self.img.forest)

        dimensions_extractors = { CodingDimension(dim): (extractor, [])
                                  for dim, extractor in enumerate(get_domain_dimension_extractors()) 
                                  if CodingDimension(dim) in self.entropy_coding_dimensions}
        while queue:
            cur = queue.popleft()
            if cur.is_leaf():
                for extractor, target in dimensions_extractors.values():
                    target.append(extractor(cur.domain))
            else:
                queue.extend(cur.children)

        return { dim: values for dim, (_, values) in dimensions_extractors.items() }
        
    def _write_header(self, buff: BitBuffer):
        header = struct.pack(HEADER_STRUCT_FMT, QUADTREE_MAGIC, self.img.info.max_scale,
                    self.img.info.scale_bits, self.img.info.offset_bits, 
                    self.img.info.img_width, self.img.info.img_height)
        self.file.write(header)

        buff.write(self.entropy_processor.processor_id().value, ENTROPY_PROCESSOR_ID_BITS)
        for dim in CodingDimension:
            if dim in self.entropy_coding_dimensions:
                buff.write(DIMIENSION_PRESENT, 1)
            else:
                buff.write(DIMENSION_MISSING, 1)
        for dim in CodingDimension:
            if dim in self.entropy_coding_dimensions:
                self.entropy_processor.serialize_dimension_coding_structures(buff, dim)

    def _write_data(self, buff: BitBuffer):
        for root in self.img.forest:
            queue = deque[QuadtreeNode]()
            queue.append(root)
            while queue:
                cur = queue.popleft()
                if cur.is_leaf():
                    dom = cur.domain
                    buff.write(LEAF_BIT, 1)
                    buff.write(*self.entropy_processor.encode(dom.start_i, CodingDimension.DOMAIN_ROW))
                    buff.write(*self.entropy_processor.encode(dom.start_j, CodingDimension.DOMAIN_COL))
                    buff.write(dom.orientation, 1)
                    buff.write(dom.rotation, 2)
                    buff.write(*self.entropy_processor.encode(dom.quantized_scale, CodingDimension.SCALE))
                    buff.write(*self.entropy_processor.encode(dom.quantized_offset, CodingDimension.OFFSET))
                else:
                    buff.write(INTERNAL_NODE_BIT, 1)
                    queue.extend(cur.children)
        if not buff.is_empty():
            buff.flush()

class QuadtreeDeserializer:
    def __init__(self) -> None:
        self.file: BinaryIO = None
        self.img: QuadtreeImage = None
        self.entropy_processor: EntropyProcessor = None
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
            buff = BitBuffer(self.file, STRUCT_BYTE_ORDER)
            self._read_header(buff) 
            self._read_data(buff) 
            return self.img
        finally:
            if should_close:
                try:
                    self.file.close()
                except:
                    pass
    
    def _read_header(self, buff: BitBuffer):
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
            img_height=header[5],
        )
        
        entropy_processor_id = buff.read(ENTROPY_PROCESSOR_ID_BITS)
        entropy_coding_dimensions = set()
        for dim in CodingDimension:
            if buff.read(1) == DIMIENSION_PRESENT:
                entropy_coding_dimensions.add(dim)
        self.entropy_processor = entropy_processor_factory(EntropyProcessorId(entropy_processor_id))
        self.entropy_processor.prepare_for_decoding(get_dimension_bit_widths(self.img.info))
        for dim in CodingDimension:
            if dim in entropy_coding_dimensions:
                self.entropy_processor.deserialize_dimension_coding_structures(buff, dim)

    def _read_data(self, buff: BitBuffer):
        self.img.forest = []
        cur_root = None
        parent_queue = deque[tuple[QuadtreeNode, int]]()

        while (node_type := buff.read(1)) != EOF:
            if node_type == LEAF_BIT:
                start_i = self.entropy_processor.decode(buff, CodingDimension.DOMAIN_ROW)
                start_j = self.entropy_processor.decode(buff, CodingDimension.DOMAIN_COL)
                orientation = buff.read(1)
                rotation = buff.read(2)
                quantized_scale = self.entropy_processor.decode(buff, CodingDimension.SCALE)
                quantized_offset = self.entropy_processor.decode(buff, CodingDimension.OFFSET)
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
