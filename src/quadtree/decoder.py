from collections import deque

import numpy as np

from quadtree.common import (MAX_GRAY, TRANSPOSED_ORIENTATION, QuadtreeImage,
                             QuadtreeNode)
from utils import average_subsample

AUTO = -1

class QuadtreeDecoder:
    def __init__(self, iterations: int = AUTO, stop_on_relative_error: float = 5e-3,
               min_iterations=2, max_iterations=10, log_stop=False) -> None:
        self.iterations = iterations
        self.stop_on_relative_error = stop_on_relative_error
        self.min_iterations = min_iterations
        self.max_iterations = max_iterations
        self.log_stop = log_stop

        self.img: np.ndarray = None
        self.next_img: np.ndarray = None

        self.use_quantized_values = True

    def using_not_quantized_values(self) -> "QuadtreeDecoder":
        self.use_quantized_values = False
        return self

    def decode(self, quadtree_img: QuadtreeImage) -> np.ndarray:
        height, width = quadtree_img.info.img_height, quadtree_img.info.img_width
        self.img = np.zeros((height, width), dtype=np.float64) 
        self.next_img = np.empty_like(self.img)
        
        if self.iterations == AUTO:
            for _ in range(self.min_iterations):
                self._de_partition(quadtree_img, 0, 0, width, height, 0)
                self.img[:] = self.next_img[:]
            for i in range(self.min_iterations, self.max_iterations):
                self._de_partition(quadtree_img, 0, 0, width, height, 0)
                err = np.linalg.norm(self.next_img - self.img, ord='fro')
                std = np.linalg.norm(self.next_img)
                self.img[:] = self.next_img[:]
                if err / std < self.stop_on_relative_error:
                    # print(f'{err}\t{stop_on_relative_error}\t{err-stop_on_relative_error}')
                    if self.log_stop:
                        print(f'Auto decoding stopped after {i} iterations')
                    break
        else:
            for _ in range(self.iterations):
                self._de_partition(quadtree_img, 0, 0, width, height, 0)
                self.img[:] = self.next_img[:]
        
        return self.img.clip(0., 255.).astype(np.uint8)
    
    def _de_partition(self, quadtree_img: QuadtreeImage, i: int, j: int, width: int, height: int, root_counter: int) -> int:
        max_square_exponent = int(min(np.log2(width), np.log2(height)))
        max_square_side = 2**max_square_exponent

        self._decode_pow2_square(quadtree_img, i, j, max_square_exponent, quadtree_img.forest[root_counter])
        root_counter += 1

        if max_square_side < width:
            root_counter = self._de_partition(quadtree_img, i, j + max_square_side, width - max_square_side, height, root_counter)
        if max_square_side < height:
            root_counter = self._de_partition(quadtree_img, i + max_square_side, j, max_square_side, height - max_square_side, root_counter)
        
        return root_counter
        
    def _decode_pow2_square(self, quadtree_img: QuadtreeImage, i: int, j: int, side_exponent: int, root: QuadtreeNode):
        queue = deque([(root, 0, i, j)])
        max_encoded_scale = np.float64(2**quadtree_img.info.scale_bits)
        max_encoded_offset = np.float64(2**quadtree_img.info.offset_bits)
        max_scale = quadtree_img.info.max_scale

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
