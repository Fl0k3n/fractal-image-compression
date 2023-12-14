from abc import ABC, abstractmethod
from collections import deque

import numpy as np

from quadtree.common import QuadtreeImage, QuadtreeNode


class QuadtreePostprocessor(ABC):
    @abstractmethod
    def postprocess(self, decoded_img: np.ndarray, quadtree_img: QuadtreeImage) -> np.ndarray:
        pass


class DeblockingQuadtreePostprocessor(QuadtreePostprocessor):
    def __init__(self) -> None:
        self.own_weight_max_depth = 5. / 6.
        self.neigh_weight_max_depth = 1. - self.own_weight_max_depth
        self.own_weight_not_max_depth = 2. / 3.
        self.neigh_weight_not_max_depth = 1. - self.own_weight_not_max_depth
        self.max_depth_range_size = 4 # TODO compute it before deblocking
        self.res = None

    def postprocess(self, decoded_img: np.ndarray, quadtree_img: QuadtreeImage) -> np.ndarray:
        self.res = np.copy(decoded_img)
        self._de_partition(quadtree_img, decoded_img, 0, 0, quadtree_img.info.img_width, quadtree_img.info.img_height, 0)
        return self.res
    
    def _de_partition(self, quadtree_img: QuadtreeImage, img: np.ndarray, i: int, j: int, width: int, height: int, root_counter: int) -> int:
        max_square_exponent = int(min(np.log2(width), np.log2(height)))
        max_square_side = 2**max_square_exponent

        self._deblock(img, i, j, max_square_exponent, quadtree_img.forest[root_counter])
        root_counter += 1

        if max_square_side < width:
            root_counter = self._de_partition(quadtree_img, img, i, j + max_square_side, width - max_square_side, height, root_counter)
        if max_square_side < height:
            root_counter = self._de_partition(quadtree_img, img, i + max_square_side, j, max_square_side, height - max_square_side, root_counter)
        
        return root_counter
  
    def _deblock(self, img: np.ndarray, i: int, j: int, side_exponent: int, root: QuadtreeNode):
        queue = deque([(root, 0, i, j)])

        while queue:
            cur, depth, range_i, range_j = queue.popleft()
            size = 2**(side_exponent - depth)

            if cur.is_leaf():
                if size <= self.max_depth_range_size:
                    w1 = self.own_weight_max_depth
                    w2 = self.neigh_weight_max_depth
                else:
                    w1 = self.own_weight_not_max_depth
                    w2 = self.neigh_weight_not_max_depth

                if range_i - 1 >= 0: 
                    self.res[range_i, range_j:range_j+size] = w1 * img[range_i, range_j:range_j+size] + w2 * img[range_i-1, range_j:range_j+size]
                if range_i + size + 1 < img.shape[0]:
                    self.res[range_i+size, range_j:range_j+size] = w1 * img[range_i+size, range_j:range_j+size] + w2 * img[range_i+size+1, range_j:range_j+size]
                if range_j - 1 >= 0:
                    self.res[range_i:range_i+size, range_j] = w1 * img[range_i:range_i+size, range_j] + w2 * img[range_i:range_i+size, range_j-1]
                if range_j + size + 1 < img.shape[1]:
                    self.res[range_i:range_i+size, range_j+size] = w1 * img[range_i:range_i+size, range_j+size] + w2 * img[range_i:range_i+size, range_j+size+1]
            else:
                newsize = size // 2
                queue.append((cur.children[0], depth + 1, range_i, range_j))
                queue.append((cur.children[1], depth + 1, range_i, range_j + newsize))
                queue.append((cur.children[2], depth + 1, range_i + newsize, range_j))
                queue.append((cur.children[3], depth + 1, range_i + newsize, range_j + newsize))
