import struct
from collections import deque
from typing import BinaryIO

from bitbuffer import EOF, BitBuffer
from entropy.processor import (ENTROPY_PROCESSOR_ID_BITS, CodingDimension,
                               EntropyProcessor, EntropyProcessorId,
                               entropy_processor_factory)
from hvv2.common import Domain, HVImage, HVNode, SplitInfo
from quadtree.common import (EncodingInfo, get_dimension_bit_widths,
                             get_domain_dimension_extractors)

HV_MAGIC = "hv".encode(encoding='ascii')
INTERNAL_NODE_BIT = 0
LEAF_BIT = 1

STRUCT_BYTE_ORDER = "!"
HEADER_STRUCT_FMT = f'{STRUCT_BYTE_ORDER}{len(HV_MAGIC)}sfBBHHI'
HEADER_SIZE = len(HV_MAGIC) + 14
SIZE_OFFSET = HEADER_SIZE - 4
UNSUPPORTED = 0.

DIMIENSION_PRESENT = 1
DIMENSION_MISSING = 0

class HVSerializer:
    def __init__(self, entropy_processor: EntropyProcessor,
                entropy_coding_dimensions: set[CodingDimension],
                required_entropy_coding_gain=1) -> None:
        self.entropy_processor = entropy_processor
        self.entropy_coding_dimensions = entropy_coding_dimensions
        self.required_entropy_coding_gain = required_entropy_coding_gain
        self.file: BinaryIO = None 

    def serialize(self, hv_img: HVImage, output: str | BinaryIO):
        self.img = hv_img
        self._init_entropy_processor()
        self.file = output
        should_close = False
        try:
            if isinstance(output, str):
                should_close = True
                self.file = open(output, 'wb')
            initial_pos = self.file.tell()
            buff = BitBuffer(self.file, STRUCT_BYTE_ORDER)
            self._write_header(buff) 
            self._write_data(buff) 
            self._write_size(initial_pos)
        finally:
            if should_close:
                try:
                    self.file.close()
                except:
                    pass

    def _init_entropy_processor(self):
        encoding_dimension_values = self._extract_all_encoding_dimension_values()
        dimension_default_bit_widths = get_dimension_bit_widths(self.img.info)
        encoding_gains = self.entropy_processor.prepare_for_encoding(encoding_dimension_values, dimension_default_bit_widths)
        self.entropy_coding_dimensions = set([dim for dim, gain in encoding_gains.items() if gain >= self.required_entropy_coding_gain]) 

    def _extract_all_encoding_dimension_values(self) -> dict[CodingDimension, list[int]]:
        if not self.entropy_coding_dimensions:
            return {}
        queue = deque[HVNode]()
        queue.append(self.img.root)

        dimensions_extractors = { CodingDimension(dim): (extractor, [])
                                  for dim, extractor in enumerate(get_domain_dimension_extractors()) 
                                  if CodingDimension(dim) in self.entropy_coding_dimensions}
        while queue:
            cur = queue.popleft()
            if cur.is_leaf():
                for extractor, target in dimensions_extractors.values():
                    target.append(extractor(cur.domain))
            else:
                queue.append(cur.c1)
                queue.append(cur.c2)

        return { dim: values for dim, (_, values) in dimensions_extractors.items() }
        
    def _write_header(self, buff: BitBuffer):
        header = struct.pack(HEADER_STRUCT_FMT, HV_MAGIC, self.img.info.max_scale,
                    self.img.info.scale_bits, self.img.info.offset_bits, 
                    self.img.info.img_width, self.img.info.img_height, 0)
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

    def _write_size(self, initial_pos: int):
        end_pos = self.file.tell()
        size = end_pos - initial_pos
        self.file.seek(initial_pos + SIZE_OFFSET)
        self.file.write(struct.pack(f"{STRUCT_BYTE_ORDER}I", size))
        self.file.seek(end_pos)

    def _write_data(self, buff: BitBuffer):
        queue = deque[HVNode]()
        queue.append(self.img.root)
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
                buff.write(int(cur.split_info.vertical_split), 1)
                buff.write(cur.split_info.split_idx, 16) # TODO
                queue.append(cur.c1)
                queue.append(cur.c2)
        if not buff.is_empty():
            buff.flush()

class HVDeserializer:
    def __init__(self) -> None:
        self.file: BinaryIO = None
        self.img: HVImage = None
        self.entropy_processor: EntropyProcessor = None
        self.domain_i_bits = -1
        self.domain_j_bits = -1
        self.initial_pos = -1

    def deserialize(self, input: str | BinaryIO) -> HVImage:
        self.file = input
        should_close = False
        self.img = HVImage(None, None)
        try:
            if isinstance(input, str):
                should_close = True
                self.file = open(input, 'rb')
            self.initial_pos = self.file.tell()
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
        if header[0] != HV_MAGIC:
            raise Exception(f'Invalid file format: file is missing hv magic')

        self.img.info = EncodingInfo(
            scale_bits=header[2],
            offset_bits=header[3],
            max_scale=header[1],
            img_width=header[4],
            img_height=header[5],
        )
        self.size = header[6]
        
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
        parent_queue = deque[tuple[HVNode, int]]()

        while self.file.tell() < self.initial_pos + self.size or not buff.is_empty():
            node_type = buff.read(1)
            assert node_type != EOF, "Invalid size encoding, buffer returned EOF"
            if node_type == LEAF_BIT:
                start_i = self.entropy_processor.decode(buff, CodingDimension.DOMAIN_ROW)
                start_j = self.entropy_processor.decode(buff, CodingDimension.DOMAIN_COL)
                orientation = buff.read(1)
                rotation = buff.read(2)
                quantized_scale = self.entropy_processor.decode(buff, CodingDimension.SCALE)
                quantized_offset = self.entropy_processor.decode(buff, CodingDimension.OFFSET)
                node = HVNode.Leaf(Domain(
                    start_i, start_j, orientation, rotation, scale=UNSUPPORTED, offset=UNSUPPORTED,
                    quantized_scale=quantized_scale, quantized_offset=quantized_offset
                ))
                if parent_queue:
                    parent, child_idx = parent_queue.popleft()
                    if child_idx == 0:
                        parent.c1 = node
                    else:
                        parent.c2 = node
                elif self.img.root is None:
                    self.img.root = node
            else:
                vertical_split = bool(buff.read(1))
                split_idx = buff.read(16)
                node = HVNode.ParentOf(None, None, SplitInfo(vertical_split, split_idx))
                if parent_queue:
                    parent, child_idx = parent_queue.popleft()
                    if child_idx == 0:
                        parent.c1 = node
                    else:
                        parent.c2 = node
                elif self.img.root is None:
                    self.img.root = node
                parent_queue.append((node, 0))
                parent_queue.append((node, 1))
