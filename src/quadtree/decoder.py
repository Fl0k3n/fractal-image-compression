from collections import deque

import numpy as np

from quadtree.encoder import TRANSPOSED_ORIENTATION, QuadtreeNode
from utils import average_subsample


class QuadtreeDecoder:
    def __init__(self) -> None:
        pass

    def decode(self, height: int, width: int, tree: list[QuadtreeNode], iterations: int = 10):
        img = np.zeros((height, width), dtype=np.float32) 
        next_img = np.empty_like(img)

        max_square_vertical = int(np.log2(img.shape[0]))
        max_square_horizontal = int(np.log2(img.shape[1]))

        img_square_exponent = min(max_square_vertical, max_square_horizontal)
        self.max_square_exponent = img_square_exponent

        # TODO remaining roots
        root = tree[0]

        for _ in range(iterations):
            queue = deque([(root, 0, 0, 0)])
            while queue:
                cur, depth, range_i, range_j = queue.popleft()

                size = 2**(img_square_exponent - depth)
                if cur.is_leaf():
                    dom = cur.domain

                    dom_i = np.copy(img[dom.start_i:dom.start_i + size * 2, dom.start_j:dom.start_j + size * 2])
                    if dom.orientation == TRANSPOSED_ORIENTATION:
                        dom_i = dom_i.T
                    dom_i = np.rot90(dom_i, dom.rotation)
                    dom_i = average_subsample(dom_i) * dom.s + dom.o

                    next_img[range_i:range_i + size, range_j:range_j + size] = dom_i
                else:
                    newsize = size // 2
                    queue.append((cur.children[0], depth + 1, range_i, range_j))
                    queue.append((cur.children[1], depth + 1, range_i, range_j + newsize))
                    queue.append((cur.children[2], depth + 1, range_i + newsize, range_j))
                    queue.append((cur.children[3], depth + 1, range_i + newsize, range_j + newsize))
                    
            img = next_img
        
        return img.astype(np.uint8)