from collections import deque

import numpy as np

from quadtree.encoder import TRANSPOSED_ORIENTATION, QuadtreeNode
from utils import average_subsample


class QuadtreeDecoder:
    def __init__(self, width: int, height: int, forest: list[QuadtreeNode]) -> None:
        self.width = width
        self.height = height
        self.forest = forest

        self.img: np.ndarray = None
        self.next_img: np.ndarray = None

    def decode(self, iterations: int = 10):
        self.img = np.zeros((self.height, self.width), dtype=np.float32) 
        self.next_img = np.empty_like(self.img)

        for _ in range(iterations):
            self._de_partition(0, 0, self.width, self.height, 0)
            self.img = self.next_img
        
        return self.img.astype(np.uint8)
    
    def _de_partition(self, i: int, j: int, width: int, height: int, root_counter: int) -> int:
        max_square_exponent = int(min(np.log2(width), np.log2(height)))
        max_square_side = 2**max_square_exponent

        self._decode_pow2_square(i, j, max_square_exponent, self.forest[root_counter])
        root_counter += 1

        if max_square_side < width:
            root_counter = self._de_partition(i, j + max_square_side, width - max_square_side, height, root_counter)
        if max_square_side < height:
            root_counter = self._de_partition(i + max_square_side, j, max_square_side, height - max_square_side, root_counter)
        
        return root_counter
        
    def _decode_pow2_square(self, i: int, j: int, side_exponent: int, root: QuadtreeNode):
        queue = deque([(root, 0, i, j)])
        while queue:
            cur, depth, range_i, range_j = queue.popleft()

            size = 2**(side_exponent - depth)
            if cur.is_leaf():
                dom = cur.domain

                dom_i = np.copy(self.img[dom.start_i:dom.start_i + size * 2, dom.start_j:dom.start_j + size * 2])
                if dom.orientation == TRANSPOSED_ORIENTATION:
                    dom_i = dom_i.T
                dom_i = np.rot90(dom_i, dom.rotation)
                dom_i = average_subsample(dom_i) * dom.scale + dom.offset

                self.next_img[range_i:range_i + size, range_j:range_j + size] = dom_i
            else:
                newsize = size // 2
                queue.append((cur.children[0], depth + 1, range_i, range_j))
                queue.append((cur.children[1], depth + 1, range_i, range_j + newsize))
                queue.append((cur.children[2], depth + 1, range_i + newsize, range_j))
                queue.append((cur.children[3], depth + 1, range_i + newsize, range_j + newsize))
