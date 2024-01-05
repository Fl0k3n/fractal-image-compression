from collections import deque

import numpy as np

from hvv2.common import HVImage, HVNode
from quadtree.common import MAX_GRAY
from quadtree.decoder import AUTO
from utils import average_subsample_jit


class HVDecoder:
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

    def using_not_quantized_values(self) -> "HVDecoder":
        self.use_quantized_values = False
        return self

    def decode(self, hv_img: HVImage) -> np.ndarray:
        height, width = hv_img.info.img_height, hv_img.info.img_width
        self.img = np.zeros((height, width), dtype=np.float64) 
        self.next_img = np.empty_like(self.img)

        if self.iterations == AUTO:
            for _ in range(self.min_iterations):
                self._decode(hv_img)
                self.img[:] = self.next_img[:]
            for i in range(self.min_iterations, self.max_iterations):
                self._decode(hv_img)
                err = np.linalg.norm(self.next_img - self.img, ord='fro')
                std = np.linalg.norm(self.next_img)
                self.img[:] = self.next_img[:]
                if err / std < self.stop_on_relative_error:
                    if self.log_stop:
                        print(f'Auto decoding stopped after {i} iterations')
                    break
        else:
            for _ in range(self.iterations):
                self._decode(hv_img)
                self.img[:] = self.next_img[:]
        return self.img.clip(0., 255.).astype(np.uint8)

    def _decode(self, hv_img: HVImage):
        q = deque[tuple[HVNode, int, int, int, int]]()
        max_encoded_scale = np.float64(2**hv_img.info.scale_bits)
        max_encoded_offset = np.float64(2**hv_img.info.offset_bits)
        max_scale = hv_img.info.max_scale

        q.append((hv_img.root, 0, 0, hv_img.info.img_width, hv_img.info.img_height))
        while q:
            node, range_i, range_j, width, height = q.popleft()
            if node.is_leaf():
                dom = node.domain

                if self.use_quantized_values:
                    scale = np.float64(dom.quantized_scale) * (2 * max_scale) / max_encoded_scale - max_scale
                    mul_coef = float(MAX_GRAY)
                    if scale < 0:
                        mul_coef *= (1 + scale)
                    offset = np.float64(dom.quantized_offset) / max_encoded_offset * mul_coef - MAX_GRAY * scale
                else:
                    scale = dom.scale
                    offset = dom.offset

                dom_i = np.copy(self.img[dom.start_i:dom.start_i + height * 2, dom.start_j:dom.start_j + width * 2])
                dom_i = average_subsample_jit(dom_i) * scale + offset

                self.next_img[range_i:range_i + height, range_j:range_j + width] = dom_i
            else:
                split_idx = node.split_info.split_idx
                if node.split_info.vertical_split:
                    q.append((node.c1, range_i, range_j, split_idx + 1, height))
                    q.append((node.c2, range_i, range_j + split_idx + 1, width - split_idx - 1, height))
                else:
                    q.append((node.c1, range_i, range_j, width, split_idx + 1))
                    q.append((node.c2, range_i + split_idx + 1, range_j, width, height - split_idx - 1))
