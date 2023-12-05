from collections import deque

import numpy as np

from quadtree.common import (MAX_GRAY, TRANSPOSED_ORIENTATION, QuadtreeImage,
                             QuadtreeNode)
from utils import average_subsample

AUTO = -1

class QuadtreeDecoder:
    def __init__(self, img: QuadtreeImage) -> None:
        self.width = img.info.img_width
        self.height = img.info.img_height
        self.forest = img.forest
        self.encoding_info = img.info

        self.img: np.ndarray = None
        self.next_img: np.ndarray = None

        self.use_quantized_values = True

    def using_not_quantized_values(self) -> "QuadtreeDecoder":
        self.use_quantized_values = False
        return self

    def decode(self, iterations: int = AUTO, stop_on_relative_error: float = 5e-3,
               min_iterations=2, max_iterations=10, log_stop=False) -> np.ndarray:
        self.img = np.zeros((self.height, self.width), dtype=np.float32) 
        self.next_img = np.empty_like(self.img)
        
        if iterations == AUTO:
            for _ in range(min_iterations):
                self._de_partition(0, 0, self.width, self.height, 0)
                self.img[:] = self.next_img[:]
            for i in range(min_iterations, max_iterations):
                self._de_partition(0, 0, self.width, self.height, 0)
                err = np.linalg.norm(self.next_img - self.img, ord='fro')
                std = np.linalg.norm(self.next_img)
                self.img[:] = self.next_img[:]
                if err / std < stop_on_relative_error:
                    # print(f'{err}\t{stop_on_relative_error}\t{err-stop_on_relative_error}')
                    if log_stop:
                        print(f'Auto decoding stopped after {i} iterations')
                    break
        else:
            for _ in range(iterations):
                self._de_partition(0, 0, self.width, self.height, 0)
                self.img[:] = self.next_img[:]
        
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
        max_encoded_scale = np.float64(2**self.encoding_info.scale_bits)
        max_encoded_offset = np.float64(2**self.encoding_info.offset_bits)
        max_scale = self.encoding_info.max_scale

        while queue:
            cur, depth, range_i, range_j = queue.popleft()

            size = 2**(side_exponent - depth)
            if cur.is_leaf():
                dom = cur.domain

                if self.use_quantized_values:
                    scale = np.float64(dom.quantized_scale) * (2 * max_scale) / max_encoded_scale - max_scale
                    mul_coef = float(MAX_GRAY)
                    if scale < 0:
                        mul_coef *= (1 + scale)
                    offset = np.float64(dom.quantized_offset) / max_encoded_offset * mul_coef - MAX_GRAY * scale
                else:
                    scale = dom.scale
                    offset = dom.offset

                dom_i = np.copy(self.img[dom.start_i:dom.start_i + size * 2, dom.start_j:dom.start_j + size * 2])

                if dom.orientation == TRANSPOSED_ORIENTATION:
                    dom_i = dom_i.T
                dom_i = np.rot90(dom_i, dom.rotation)
                dom_i = average_subsample(dom_i) * scale + offset

                self.next_img[range_i:range_i + size, range_j:range_j + size] = dom_i
            else:
                newsize = size // 2
                queue.append((cur.children[0], depth + 1, range_i, range_j))
                queue.append((cur.children[1], depth + 1, range_i, range_j + newsize))
                queue.append((cur.children[2], depth + 1, range_i + newsize, range_j))
                queue.append((cur.children[3], depth + 1, range_i + newsize, range_j + newsize))
