import heapq
from collections import Counter, deque
from dataclasses import dataclass, field
from typing import Optional

from bitbuffer import BitBuffer
from entropy.abccodec import Code, CodeWidth, EntropyEncoder

LEFT_BIT = 0
RIGHT_BIT = 1

INTERNAL_NODE_BIT = 0
LEAF_BIT = 1

UNAVAILABLE = -1
TABLE_VAL_BITS_ENCODING_SIZE_BITS = 8

CodingTable = dict[int, tuple[Code, CodeWidth]]

@dataclass(order=True)
class HuffmanNode:
    freq: int
    val: Optional[int] = field(compare=False)
    left: Optional["HuffmanNode"] = field(compare=False)
    right: Optional["HuffmanNode"] = field(compare=False)

    @staticmethod
    def Leaf(val: int, freq: int) -> "HuffmanNode":
        return HuffmanNode(val=val, freq=freq, left=None, right=None)
    
    @staticmethod
    def Inner(left: "HuffmanNode", right: "HuffmanNode") -> "HuffmanNode":
        return HuffmanNode(val=None, freq=left.freq + right.freq, left=left, right=right)
    
    def is_leaf(self) -> bool:
        return self.left is None # left is None <=> right is None

class HuffmanEncoder(EntropyEncoder):
    def __init__(self) -> None:
        self.root: HuffmanNode = None
        self.coding_table: CodingTable = None

    def prepare_for_encoding(self, symbols: list[int]):
        heap = [HuffmanNode.Leaf(symbol, freq) for symbol, freq in Counter(symbols).items()]
        heapq.heapify(heap)
        while len(heap) > 1:
            n1 = heapq.heappop(heap)
            n2 = heapq.heappop(heap)
            heapq.heappush(heap, HuffmanNode.Inner(n1, n2))

        self.root = heap[0]
        self.coding_table = self._build_coding_table(self.root)

    def encode(self, symbol: int):
        return self.coding_table[symbol]
    
    def _build_coding_table(self, root: HuffmanNode) -> CodingTable:
        coding_tab: CodingTable = {}
                
        def traverse_and_write_codes(cur: HuffmanNode, cur_code: int, depth: int):
            if cur.is_leaf():
                coding_tab[cur.val] = (cur_code, depth)
            else:
                traverse_and_write_codes(cur.left, (cur_code << 1) + LEFT_BIT, depth + 1)
                traverse_and_write_codes(cur.right, (cur_code << 1) + RIGHT_BIT, depth + 1)
        
        traverse_and_write_codes(root, 0, 0)
        return coding_tab
    
class HuffmanDecoder:
    def __init__(self, root: HuffmanNode) -> None:
        self.root = root

    def decode(self, read_buff: BitBuffer) -> int:
        cur = self.root
        while True:
            if cur.is_leaf():
                return cur.val
            else:
                cur = cur.left if read_buff.read(1) == LEFT_BIT else cur.right
            
class HuffmanSerializer:
    def serialize(self, root: HuffmanNode, val_bits: int, write_buff: BitBuffer):
        assert not root.is_leaf()
        queue = deque[HuffmanNode]()
        queue.append(root.left)
        queue.append(root.right)

        while queue: 
            cur = queue.popleft()
            if cur.is_leaf():
                write_buff.write(LEAF_BIT, 1)
                write_buff.write(cur.val, val_bits)
            else:
                write_buff.write(INTERNAL_NODE_BIT, 1)
                queue.append(cur.left)
                queue.append(cur.right)

class HuffmanDeserializer:
    def deserialize(self, read_buff: BitBuffer, val_bits: int) -> HuffmanNode:
        parent_queue = deque[HuffmanNode]()
        root = HuffmanNode(UNAVAILABLE, None, None, None)
        parent_queue.append(root)
        parent_queue.append(root)

        while parent_queue:
            parent = parent_queue.popleft()
            if read_buff.read(1) == LEAF_BIT:
                cur = HuffmanNode.Leaf(read_buff.read(val_bits), UNAVAILABLE)
            else:
                cur = HuffmanNode(UNAVAILABLE, None, None, None)
                parent_queue.append(cur)
                parent_queue.append(cur)
            if parent.left is None:
                parent.left = cur
            else:
                parent.right = cur

        return root
                    
if __name__ == "__main__":
    symbols = [5, 5, 5, 5, 5, 1, 2, 3, 1, 1, 2, 2, 3]
    encoder = HuffmanEncoder()
    encoder.prepare_for_encoding(symbols)
    encoded = [encoder.encode(symbol) for symbol in symbols]
    def dummy_bit_gen():
        for code, width in encoded:
            for i in range(width - 1, -1, -1):
                yield (code >> i) & 1
    gen = dummy_bit_gen()
    class DummyBitBuff:
        def read(*args):
            return next(gen) 
    dbb = DummyBitBuff()

    decoder = HuffmanDecoder(encoder.root)
    decoded = [decoder.decode(dbb) for _ in encoded]
    assert len(symbols) == len(decoded) and all(sym == decoded_sym for sym, decoded_sym in zip(symbols, decoded))
    print('coding-decoding OK')

    import io
    with io.BytesIO() as bytes_io:
        bit_buff = BitBuffer(bytes_io)
        HuffmanSerializer().serialize(encoder.root, 8, bit_buff)
        if not bit_buff.is_empty():
            bit_buff.flush()
        bytes_io.flush()
        bytes_io.seek(0)
        bit_buff = BitBuffer(bytes_io)
        deserialized_root = HuffmanDeserializer().deserialize(bit_buff, 8)

    ct = HuffmanEncoder()._build_coding_table(deserialized_root)
    assert len(ct) == len(encoder.coding_table) and all(ct[k] == encoder.coding_table[k] for k in ct.keys())
    print('de-serialization OK')

