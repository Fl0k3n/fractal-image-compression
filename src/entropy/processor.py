from abc import ABC, abstractmethod
from enum import Enum

from bitbuffer import BitBuffer
from entropy.abccodec import Code, CodeWidth, EntropyEncoder, IdentityEncoder
from entropy.huffman import (HuffmanDecoder, HuffmanDeserializer,
                             HuffmanEncoder, HuffmanSerializer)


class CodingDimension(Enum):
    # must be number from 0 to n
    DOMAIN_ROW = 0
    DOMAIN_COL = 1
    SCALE = 2
    OFFSET = 3

ENTROPY_PROCESSOR_ID_BITS = 8

class EntropyProcessorId(Enum):
    HUFFMAN = 1

class EntropyProcessor(ABC):
    @abstractmethod
    def prepare_for_encoding(self, dimensions: dict[CodingDimension, list[int]], dimension_default_bit_widths: list[int]):
        """Prepares processor for encoding.

        Args:
            dimensions (dict[CodingDimension, list[int]]): dimensions that should be subject to entropy coding
            dimension_default_bit_widths (list[int]): default bit widths of all dimensions, including optional ones that 
            shouldn't be entropy-coded
        """
        pass

    @abstractmethod
    def serialize_dimension_coding_structures(self, write_buff: BitBuffer, dimension: CodingDimension):
        pass

    @abstractmethod
    def encode(self, value: int, dimension: CodingDimension) -> tuple[Code, CodeWidth]:
        """Entropy-encodes given value using encoder for given dimension. If dimension shouldn't be
        entropy-encoded identity value is returned with default bit width."""
        pass

    @abstractmethod
    def prepare_for_decoding(self, dimension_default_bit_widths: list[int]):
        pass

    @abstractmethod
    def deserialize_dimension_coding_structures(self, read_buff: BitBuffer, dimension: CodingDimension):
        pass
        
    @abstractmethod
    def decode(self, read_buff: BitBuffer, dimension: CodingDimension) -> int:
        pass

    @abstractmethod
    def processor_id(self) -> EntropyProcessorId:
        pass

def entropy_processor_factory(id: EntropyProcessorId) -> EntropyProcessor:
    if id == EntropyProcessorId.HUFFMAN:
        return HuffmanEntropyProcessor()
    raise Exception(f"Unknown entropy processor: {id}")

class HuffmanEntropyProcessor(EntropyProcessor):
    def __init__(self) -> None:
        super().__init__()
        self.dimension_encoders: list[EntropyEncoder] = [None for _ in CodingDimension]
        self.dimension_decoders: list[HuffmanDecoder] = [None for _ in CodingDimension]
        self.dimension_default_bit_widths: list[int]  = [-1 for _ in CodingDimension]

    def prepare_for_encoding(self, dimensions: dict[CodingDimension, list[int]], dimension_default_bit_widths: list[int]):
        self.dimension_default_bit_widths = dimension_default_bit_widths
        assert len(dimension_default_bit_widths) == len(self.dimension_encoders), \
            "Every dimension requires default bit width"

        for dimension, values in dimensions.items():
            encoder = HuffmanEncoder()
            encoder.prepare_for_encoding(values)
            self.dimension_encoders[dimension.value] = encoder

        for i, bit_width in enumerate(self.dimension_default_bit_widths):
            if self.dimension_encoders[i] is None:
                self.dimension_encoders[i] = IdentityEncoder(bit_width)

    def serialize_dimension_coding_structures(self, write_buff: BitBuffer, dimension: CodingDimension):
        bit_width = self.dimension_default_bit_widths[dimension.value]
        encoder = self.dimension_encoders[dimension.value]
        assert isinstance(encoder, HuffmanEncoder)
        HuffmanSerializer().serialize(encoder.root, bit_width, write_buff)

    def encode(self, value: int, dimension: CodingDimension) -> tuple[Code, CodeWidth]:
        return self.dimension_encoders[dimension.value].encode(value)
    
    def prepare_for_decoding(self, dimension_default_bit_widths: list[int]):
        self.dimension_default_bit_widths = dimension_default_bit_widths

    def deserialize_dimension_coding_structures(self, read_buff: BitBuffer, dimension: CodingDimension):
        root = HuffmanDeserializer().deserialize(read_buff, self.dimension_default_bit_widths[dimension.value])
        self.dimension_decoders[dimension.value] = HuffmanDecoder(root)
        
    def decode(self, read_buff: BitBuffer, dimension: CodingDimension) -> int:
        decoder = self.dimension_decoders[dimension.value]
        if decoder is not None:
            return decoder.decode(read_buff)
        return read_buff.read(self.dimension_default_bit_widths[dimension.value])

    def processor_id(self) -> EntropyProcessorId:
        return EntropyProcessorId.HUFFMAN
    