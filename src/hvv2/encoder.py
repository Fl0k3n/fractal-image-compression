import numpy as np
# from numba import njit

from hvv2.common import Domain, HVImage, HVNode, SplitInfo
from quadtree.common import MAX_GRAY, EncodingInfo
from quadtree.encoder import ZERO_TOLERANCE
from utils import average_subsample_jit


class HVEncoder:
    def __init__(self, min_range_height: int, min_range_width: int,
                max_range_height: int, max_range_width: int,
                scale_bits: int, offset_bits: int, err_threshold: float,
                max_scale: float=1.0) -> None:
        self.min_range_height = min_range_height
        self.min_range_width = min_range_width
        self.max_range_height = max_range_height
        self.max_range_width = max_range_width
        self.err_threshold = err_threshold
        self.scale_bits = scale_bits
        self.offset_bits = offset_bits
        self.max_scale = max_scale

    def encode(self, img: np.ndarray) -> HVImage:
        root = self._partition(img.astype(np.int32), 0, 0, img.shape[1], img.shape[0])
        info = EncodingInfo(
            scale_bits=self.scale_bits,
            offset_bits=self.offset_bits,
            max_scale=self.max_scale,
            img_width=img.shape[1],
            img_height=img.shape[0]
        )
        return HVImage(info=info, root=root)

    def _partition(self, img: np.ndarray, i: int, j: int, width: int, height: int) -> HVNode:
        if width <= self.min_range_width and height <= self.min_range_height:
            dom, _ = self._find_optimal_domain(img, i, j, width, height) 
            return HVNode.Leaf(dom)
        
        if width <= self.max_range_width and height <= self.max_range_height:
            dom, err = self._find_optimal_domain(img, i, j, width, height) 
            if err < self.err_threshold:
                return HVNode.Leaf(dom)

        best_vertical_split, best_vertical_split_diff = None, None
        best_horizontal_split, best_horizontal_split_diff = None, None
        
        if width > self.min_range_width:
            col_sum_deltas = self._compute_biased_sum_deltas(img, i, j, width, height, 0)
            best_vertical_split = np.argmax(col_sum_deltas)
            best_vertical_split_diff = col_sum_deltas[best_vertical_split]
        if height > self.min_range_height:
            row_sum_deltas = self._compute_biased_sum_deltas(img, i, j, width, height, 1)
            best_horizontal_split = np.argmax(row_sum_deltas)
            best_horizontal_split_diff = row_sum_deltas[best_horizontal_split]

        vertical_split = False
        if best_vertical_split is None:
            vertical_split = False
        elif best_horizontal_split is None:
            vertical_split = True
        else:
            vertical_split = np.abs(best_vertical_split_diff) > np.abs(best_horizontal_split_diff) 

        child1, child2, split_idx = None, None, None
        if vertical_split:
            child1 = self._partition(img, i, j, best_vertical_split + 1, height)
            child2 = self._partition(img, i, j + best_vertical_split + 1, width - best_vertical_split - 1, height)
            split_idx = best_vertical_split
        else:
            child1 = self._partition(img, i, j, width, best_horizontal_split + 1)
            child2 = self._partition(img, i + best_horizontal_split + 1, j, width, height - best_horizontal_split - 1)
            split_idx = best_horizontal_split

        return HVNode.ParentOf(child1, child2, SplitInfo(vertical_split, split_idx))
            
    def _compute_biased_sum_deltas(self, img: np.ndarray, i: int, j: int, width: int, height: int, axis: int) -> np.ndarray:
        sums = np.sum(img[i: i+height, j: j+width], axis=axis)
        sum_deltas = sums[:-1] - sums[1:]
        biases = np.arange(sums.shape[0] - 1, dtype=np.float64)
        biases = np.minimum(biases, sums.shape[0] - 1 - biases) / (sums.shape[0] - 1)
        return sum_deltas * biases

    def _find_optimal_domain(self, img: np.ndarray, i: int, j: int, width: int, height: int) -> tuple[Domain, float]:
        domain, err = _run_domain_search(img, i, j, width, height, self.scale_bits, self.offset_bits, self.max_scale)
        return Domain(
            start_i=domain[0],
            start_j=domain[1],
            orientation=domain[2],
            rotation=domain[3],
            scale=domain[4],
            offset=domain[5],
            quantized_scale=domain[6],
            quantized_offset=domain[7],
        ), err

# @njit
def _run_domain_search(img: np.ndarray, i: int, j: int, width: int, height: int,
        scale_bits: int, offset_bits: int, max_scale: float) -> tuple[
            tuple[int, int, int, int, np.float32, np.float32, np.float32, int, int], np.float32
        ]:
    img_range = img[i: i+height, j: j+width].astype(np.float32)
    range_sum = np.sum(img_range)
    squared_range_sum = np.sum(img_range * img_range)
    best_error = np.inf
    best_domain = (0, 0, 0, 0, 0., 0., 0, 0)

    for domain_i in range(0, img.shape[0] - 2 * height, height):
        for domain_j in range(0, img.shape[1] - 2 * width, width):
            candidate_domain = img[domain_i: domain_i+2*height, domain_j: domain_j+2*width] #.astype(np.float32)
            for orientation, oriented_domain in enumerate((candidate_domain, np.rot90(candidate_domain.T, 1))):
                for rotation in [0, 2]:
                    img_domain = np.rot90(oriented_domain, rotation).astype(np.float32)
                    domain_subsampled = average_subsample_jit(img_domain)
                    domain_sum = np.sum(domain_subsampled)
                    squared_domain_sum = np.sum(domain_subsampled * domain_subsampled)
                    error, scale_factor, offset_factor, quantized_scale_factor, quantized_offset_factor =  _find_optimal_mean_square_error(
                        domain_subsampled, img_range, max_scale, scale_bits, offset_bits, range_sum, domain_sum,
                        squared_range_sum, squared_domain_sum, width, height)
                    
                    if error < best_error:
                        best_error = error
                        best_domain = (domain_i, domain_j, orientation, rotation,
                                    scale_factor, offset_factor, quantized_scale_factor, quantized_offset_factor)
                
    return best_domain, best_error

# @njit
def _find_optimal_mean_square_error(domain: np.ndarray, range_: np.ndarray, max_scale: np.float32,
        scale_bits: int, offset_bits: int, range_sum: np.float32, domain_sum: np.float32,
        squared_range_sum: np.float32, squared_domain_sum: np.float32, range_width: int, range_height: int
        ) -> tuple[np.float32, np.float32, np.float32, int, int]:
    n = range_width * range_height
    ab_sum = np.sum(domain * range_)
    
    denominator = (n * squared_domain_sum - domain_sum * domain_sum)
    if np.abs(denominator) < ZERO_TOLERANCE:
        s = 0.
        o = 1. / n * range_sum
    else:
        s = (n * ab_sum - range_sum * domain_sum) / denominator
        o = 1. / n * (range_sum - s * domain_sum)

    if s < -max_scale or s > max_scale:
        s = -max_scale
        o_left = 1. / n * (range_sum - s * domain_sum)
        R_left, qs_left, qo_left = _calc_mean_square_error(n, squared_range_sum, s, squared_domain_sum, ab_sum,
                                     o_left, domain_sum, range_sum, max_scale, scale_bits, offset_bits)
        s = max_scale
        o = 1. / n * (range_sum - s * domain_sum)
        R_right, qs_right, qo_right = _calc_mean_square_error(n, squared_range_sum, s, squared_domain_sum, ab_sum,
                                     o, domain_sum, range_sum, max_scale, scale_bits, offset_bits)
        if R_right < R_left:
            return R_right, s, o, qs_right, qo_right
        return R_left, -max_scale, o_left, qs_left, qo_left
    else:
        R, qs, qo = _calc_mean_square_error(n, squared_range_sum, s, squared_domain_sum, ab_sum,
                             o, domain_sum, range_sum, max_scale, scale_bits, offset_bits)
        return R, s, o, qs, qo
    
# @njit
def _calc_mean_square_error(n: int, squared_range_sum: np.float32, s: np.float32, squared_domain_sum: np.float32,
                       ab_sum: np.float32, o: np.float32, domain_sum: np.float32, range_sum: np.float32,
                       max_scale: float, scale_bits: int, offset_bits: int) -> tuple[float, int, int]:
    # TODO positive-only-mode can have better precision using same scale_bits
    qs = int((s + max_scale) / (2 * max_scale) * 2**scale_bits)
    if qs < 0.: qs = 0
    if qs > 2**scale_bits - 1: qs = 2**scale_bits - 1
    s = np.float64(qs) * (2 * max_scale) / np.float64(2**scale_bits) - max_scale

    offset_div_coef = np.float64(MAX_GRAY)
    if s < 0.:
        offset_div_coef *= (1 + s)
        if s == -1.:
            # zero division error may happen as we set s = -1. explicitly with max_scale = 1 if it is out of bounds
            offset_div_coef = 1e-6
    qo = int((o + MAX_GRAY * s) / offset_div_coef * 2**offset_bits)
    if qo < 0: qo = 0 
    if qo > 2**offset_bits - 1: qo = 2**offset_bits - 1
    o = np.float64(qo) / 2**offset_bits * offset_div_coef - MAX_GRAY * s

    R = 1. / n * (
        squared_range_sum +
        s * (s * squared_domain_sum - 2 * ab_sum + 2 * o * domain_sum) +
        o * (n * o - 2 * range_sum)
    )

    return R, qs, qo
