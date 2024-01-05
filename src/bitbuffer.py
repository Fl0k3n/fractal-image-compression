import struct
from typing import BinaryIO

EOF = -1

class BitBuffer:
    """Buffer that supports writing and reading unsigned integers with arbitrary bit widths.
    """
    MAX_BIT_WIDTH = 32
    MAX_BYTE_WIDTH = MAX_BIT_WIDTH // 8

    def __init__(self, backing_buff: BinaryIO, struct_byte_order="!") -> None:
        self.buff = 0
        self.accumulated_size = 0
        self.backing_buff = backing_buff
        self.struct_byte_order = struct_byte_order
    
    def write(self, v: int, bit_width: int) -> bytes | None:
        # only unsigned integral values of size <= MAX_BIT_WIDTH can be used 
        if self.accumulated_size + bit_width >= self.MAX_BIT_WIDTH:
            free_space = self.MAX_BIT_WIDTH - self.accumulated_size
            mask = (1 << free_space) - 1
            self.buff <<= free_space
            self.buff |= (mask & (v >> (bit_width - free_space)))
            
            serialized_buff = struct.pack(f"{self.struct_byte_order}I", self.buff)
            self.accumulated_size = bit_width - free_space
            mask = (1 << self.accumulated_size) - 1
            self.buff = mask & v

            self.backing_buff.write(serialized_buff)
            return serialized_buff
        else:
            mask = (1 << bit_width) - 1
            self.buff <<= bit_width
            self.buff |= (mask & v)
            self.accumulated_size += bit_width

    def read(self, bit_width: int) -> int:
        res = 0
        if self.accumulated_size < bit_width:
            bits_to_read = bit_width - self.accumulated_size
            mask = (1 << self.accumulated_size) - 1
            res = (self.buff & mask) << bits_to_read
            serialized = self.backing_buff.read(self.MAX_BYTE_WIDTH)
            if len(serialized) < self.MAX_BYTE_WIDTH:
                return EOF
            self.buff = struct.unpack(f'{self.struct_byte_order}I', serialized)[0]
            self.accumulated_size = self.MAX_BIT_WIDTH
        else:
            bits_to_read = bit_width
            
        mask = (1 << bits_to_read) - 1
        res |= ((self.buff >> (self.accumulated_size - bits_to_read)) & mask)
        self.accumulated_size -= bits_to_read
        return res

    def flush(self) -> bytes:
        self.buff <<= (self.MAX_BIT_WIDTH - self.accumulated_size)
        serialized_buff = struct.pack(f"{self.struct_byte_order}I", self.buff)
        self.buff = 0
        self.accumulated_size = 0
        self.backing_buff.write(serialized_buff)
        return serialized_buff
    
    def is_empty(self) -> bool:
        return self.accumulated_size == 0
    
            
if __name__ == "__main__":
    class Dummy:
        def __init__(self) -> None:
            self.read_buff = struct.pack('!III', 0b0110101011, 0b0111 << 28, 12)
            self.ptr = 0
        def write(*args):
            pass
        def read(self, num):
            res = self.read_buff[self.ptr:self.ptr+num]
            self.ptr += num
            return res

    bb = BitBuffer(backing_buff=Dummy())
    bb.write(346, 16)
    bb.write(200, 8)
    v1 = bb.write(3434, 16)
    v2 = bb.write(1, 24)
    v3 = bb.write(1, 32)
    bb.write(0x4FF, 11)
    v4 = bb.write(0xFFFFF, 21)
    v5 = bb.write(0xFFFFFFFF, 32)

    print(v1.hex())
    print(v2.hex())
    print(v3.hex())
    print(v4.hex())
    print(v5.hex())

    bb = BitBuffer(backing_buff=Dummy())
    print(bb.read(22))
    print(bb.read(1))
    print(bb.read(1))
    print(bb.read(1))
    print(bb.read(1))
    bb.read(5)
    print(bb.read(3))