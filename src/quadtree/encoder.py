from enum import Enum

import numpy as np
from numba import njit

from quadtree.common import Domain, EncodingInfo, QuadtreeNode
from utils import average_subsample_jit


class DomainSelectionStrategy(Enum):
    D2 = 0

NORMAL_ORIENTATION = 0
TRANSPOSED_ORIENTATION = 1

NUM_MEAN_CLASSES = 3
NUM_VARIANCE_CLASSES = 24

POSITIVE = 1
NEGATIVE = -1
ZERO_TOLERANCE = 1e-6

SAME_AS_SIDE = -1

"""
1. classify domains:
    - need an array with size (max_partition - min_partition + 1)
    - after we subdivided original image min_partition times, we obtain a range of size 2^m, the array
        above has to contain at first position domains of size 2^m * 2, then second position has domains of size 2^m etc...
    - for each domain size:
        - create array with shape (#Mean_class, #Variance_class) = (3, 24),
            each element of this array should be a linked list of domains with given class
        - insert all domains to correct lists
2. perform quadtree partitioning:
    - divide min_partition times, recursively
    - now we have a maximum valid size, classify this range and check all domains with same class, if error is
        smaller than tolerance then add this as a part of result, otherwise divide this range 4 times (if we didn't
        exceed max_partitions)
"""

class QuadtreeEncoder:
    def __init__(self, encoding_info: EncodingInfo, full_class_search: bool, only_positive_scale: bool,
                 min_range_side: int, max_range_side: int, error_tolerance: float,
                 domain_hstep: int = SAME_AS_SIDE,  domain_vstep: int = SAME_AS_SIDE) -> None:
        """
        Args:
            encoding_info (EncodingInfo): encoding metadata
            full_class_search (bool): whether every mean and variance class should be searched for each range
            min_range_side (int): don't divide sub images that reach this size
            max_range_side (int): must divide sub images until this size is reached
            error_tolerance (float): if rms error of compressed sub image is bigger that this value, sub image is divide into 4 parts
        """
        self.encoding_info = encoding_info
        self.full_class_search = full_class_search
        self.only_positive_scale = only_positive_scale
        self.min_range_exponent = int(np.log2(min_range_side))
        self.max_range_exponent = int(np.log2(max_range_side))
        self.error_tolerance = error_tolerance
        self.domain_hstep = domain_hstep
        self.domain_vstep = domain_vstep

        self.domain_lists = None
        self.max_square_exponent = None
        # if image is a rectangle we might end up with ranges with smaller side than min_range_side
        self.min_needed_range_exponent = None 

    def encode(self, img: np.ndarray) -> list[QuadtreeNode]:
        height, width = img.shape
        max_square_vertical = int(np.log2(height))
        max_square_horizontal = int(np.log2(width))
        self.max_square_exponent = min(max_square_vertical, max_square_horizontal)

        self.min_needed_range_exponent = min(self.min_range_exponent, 
                                            max(int(np.log2(self._get_smallest_range_size(width, height))), 1))
        self.domain_lists = self._build_domain_lists(img)

        quandtree_roots = self._partition_image(img, 0, 0, width, height)
        return quandtree_roots
    
    def _get_smallest_range_size(self, width: int, height: int) -> int:
        if width == 0:
            return height
        if height == 0:
            return width
        if width == 1 or height == 1:
            return 1
        max_square = 2**int(min(np.log2(width), np.log2(height)))
        if max_square == width and max_square == height:
            return max_square
        fits_on_x = width // max_square
        fits_on_y = height // max_square
        min_right = self._get_smallest_range_size(width - fits_on_x * max_square, height)
        min_bottom = self._get_smallest_range_size(fits_on_x * max_square, height - fits_on_y * max_square)
        return min(min_bottom, min_right)

    def _build_domain_lists(self, img: np.ndarray):
        min_domain_exponent = self.min_needed_range_exponent + 1
        max_domain_exponent = self.max_range_exponent + 1

        domain_lists = [None] * (max_domain_exponent - min_domain_exponent + 1)

        for i, domain_exponent in enumerate(range(min_domain_exponent, max_domain_exponent + 1)):
            side = 2**domain_exponent
            hstep = side if self.domain_hstep == SAME_AS_SIDE else self.domain_hstep
            vstep = side if self.domain_vstep == SAME_AS_SIDE else self.domain_vstep
            domain_lists[i] = _build_domain_list(img, side, hstep, vstep) 
        
        return domain_lists

    def _partition_image(self, img: np.ndarray, start_i: int, start_j: int, width: int, height: int) -> list[QuadtreeNode]:
        max_square_vertical = int(np.log2(height))
        max_square_horizontal = int(np.log2(width))
        max_square_exponent = min(max_square_vertical, max_square_horizontal)

        max_square_side = 2**max_square_exponent
        main_node = self._quadtree(img, start_i, start_j, max_square_side, max_square_exponent)
        roots = [main_node]

        if max_square_side < width:
            roots.extend(self._partition_image(img, start_i, start_j + max_square_side, width - max_square_side, height))
        if max_square_side < height:
            roots.extend(self._partition_image(img, start_i + max_square_side, start_j, max_square_side, height - max_square_side))

        return roots

    def _quadtree(self, img: np.ndarray, start_i: int, start_j: int, side: int, size_exponent: int) -> QuadtreeNode:
        """Performs quadtree over img[start_i:start_i:side, start_j:start_j+side]. Side must be power of 2."""
        if size_exponent > self.max_range_exponent:
            return self._divide_into_4_subsquares(img, start_i, start_j, side // 2, size_exponent - 1)
            
        best_domain = self._find_best_domain(img, start_i, start_j, side, size_exponent)
        # best_domain = _find_best_domain_bruteforce(img, img[start_i:start_i + side, start_j: start_j + side], side * 2)
        domain_i, domain_j, orient, rot, scale_factor, offset_factor, rms_err = best_domain
        
        if rms_err > self.error_tolerance and size_exponent > self.min_range_exponent:
            # TODO probably should check if mean error of these divided subsquares is lower than this square, is it even possible?
            return self._divide_into_4_subsquares(img, start_i, start_j, side // 2, size_exponent - 1)
        else:
            return QuadtreeNode.Leaf(Domain(domain_i, domain_j, orient, rot, scale_factor, offset_factor))

    def _divide_into_4_subsquares(self, img: np.ndarray, start_i: int, start_j: int, new_side: int, new_depth: int) -> QuadtreeNode:
        c1 = self._quadtree(img, start_i, start_j, new_side, new_depth)
        c2 = self._quadtree(img, start_i, start_j + new_side, new_side, new_depth)
        c3 = self._quadtree(img, start_i + new_side, start_j, new_side, new_depth)
        c4 = self._quadtree(img, start_i + new_side, start_j + new_side, new_side, new_depth)
        return QuadtreeNode.ParentOf(c1, c2, c3, c4)

    def _find_best_domain(self, img: np.ndarray, start_i: int, start_j: int, side: int,
            size_exponent: int) -> tuple[int, int, int, int, np.float32, np.float32, np.float32]:
        if side == 1:
            # TODO can't classify subimage with side = 1 (i.e. images with odd width/height will have black border)
            return (0, 0, 0, 0, 0., 0., 1e-4)

        domain_size_idx = size_exponent - self.min_needed_range_exponent
        best_domain = None
        best_err = np.inf

        modes = (POSITIVE,) if self.only_positive_scale else (POSITIVE, NEGATIVE)

        for mode in modes:
            if mode == POSITIVE:
                mc, vc, rot, orient = _classify(img, start_i, start_j, side)
            else:
                # for negative scale we need to apply another rot/orient operations to get into canonical form
                # these transformations should be equivalent to first rotating and orienting and then applying fixes,
                # fixes are (mc = 0 -> rot2, mc = 1 -> rot1 + T, mc = 2 -> T + rot-1)
                # if mc == 0:
                #     rot = (rot + 2) % 4
                # else:
                #     rot_update = 1 if orient == NORMAL_ORIENTATION else -1
                #     rot = (rot + rot_update) % 4
                #     orient = (orient + 1) % 2
                # TODO variance class also changes, we should probably hardcode it based on the book, for now recompute mc and vc
                mc, vc, rot, orient = _classify(-(img.astype(np.int32)), start_i, start_j, side)
                candidate_domains = self.domain_lists[domain_size_idx][mc][vc]

            if self.full_class_search:
                mcs = list(range(NUM_MEAN_CLASSES))
                vcs = list(range(NUM_VARIANCE_CLASSES))
            else:
                mcs = [mc]
                vcs = [vc]
            
            for mc in mcs:
                for vc in vcs:
                    candidate_domains = self.domain_lists[domain_size_idx][mc][vc]
                    domain_info, err = _find_min_err_domain(img, start_i, start_j, side,
                        -self.encoding_info.max_scale, self.encoding_info.max_scale, rot, orient, candidate_domains)
                    if err < best_err:
                        best_err = err
                        best_domain = domain_info

        return *best_domain, np.sqrt(err)
        
@njit
def _classify(img: np.ndarray, start_i: int, start_j: int, side: int) -> tuple[int, int, int, int]:
    """Returns mean class, variance class, rotation and orientation required to make image a member of these classes.
       Side must be power of 2.
    """
    # TODO simplify this and cache stuff that is recomputed multiple times

    halfside = side // 2
    a1 = np.mean(img[start_i:start_i + halfside, start_j:start_j + halfside])
    a2 = np.mean(img[start_i:start_i + halfside, start_j + halfside:start_j + side])
    a3 = np.mean(img[start_i + halfside:start_i + side, start_j:start_j + halfside])
    a4 = np.mean(img[start_i + halfside:start_i + side, start_j + halfside:start_j + side])
    
    v1 = np.sum(img[start_i:start_i + halfside, start_j:start_j + halfside] ** 2 - a1 ** 2)
    v2 = np.sum(img[start_i:start_i + halfside, start_j + halfside:start_j + side] ** 2 - a2 ** 2)
    v3 = np.sum(img[start_i + halfside:start_i + side, start_j:start_j + halfside] ** 2 - a3 ** 2)
    v4 = np.sum(img[start_i + halfside:start_i + side, start_j + halfside:start_j + side] ** 2 - a4 ** 2)

    means = np.array([a1, a2, a4, a3])
    variances = np.array([v1, v2, v4, v3])

    rotation = (4 -  np.argmax(means)) % 4 

    # same as if we did np.rot90(rotation) and linearized it like above
    means = np.roll(means, rotation)
    variances = np.roll(variances, rotation)
    orientation = TRANSPOSED_ORIENTATION if means[1] < means[3] else NORMAL_ORIENTATION

    if orientation == TRANSPOSED_ORIENTATION:
        means[1], means[3] = means[3], means[1]
        variances[1], variances[3] = variances[3], variances[1]
    
    A1, A2, A3, A4 = means[0], means[1], means[3], means[2]
    V1, V2, V3, V4 = variances[0], variances[1], variances[3], variances[2]

    if A2 >= A3 >= A4:
        mean_class = 0
    elif A2 >= A4 >= A3:
        mean_class = 1
    else:
        mean_class = 2

    ordered_variances = [(V1, 0), (V2, 1), (V3, 2), (V4, 3)]
    for i in range(1, 4):
        for j in range(i, 0, -1):
            if ordered_variances[j][0] > ordered_variances[j - 1][0]:
                ordered_variances[j - 1], ordered_variances[j] = ordered_variances[j], ordered_variances[j - 1]
            else:
                break

    # copied from the book
    order = [v[1] for v in ordered_variances]
    vc = 0
    for i in range(2, -1, -1):
        for j in range(i + 1):
            if order[j] > order[j + 1]:
                order[j], order[j + 1] = order[j + 1], order[j]
                if order[j] == 0 or order[j + 1] == 0:
                    vc += 6
                elif order[j] == 1 or order[j + 1] == 1:
                    vc += 2
                elif order[j] == 2 or order[j + 1] == 2:
                    vc += 1

    return mean_class, vc, rotation, orientation

def _build_domain_list(img: np.ndarray, side: int, hstep: int, vstep: int) -> np.ndarray:
    # TODO make it JITable
    n_vertical_domains = 1 + (img.shape[0] - side) // vstep
    n_horizontal_domains = 1 + (img.shape[1] - side) // hstep
    
    # domain_lists: (mc: int, vc: int) -> [(domain_i: int, domain_j: int, rot: int, orient: int)]
    domain_lists = []
    for i in range(NUM_MEAN_CLASSES):
        vc_list = [[] for _ in range(NUM_VARIANCE_CLASSES)]
        domain_lists.append(vc_list)

    for i in range(n_vertical_domains):
        for j in range(n_horizontal_domains):
            mean_class, variance_class, rotation, orientation = _classify(img, i * vstep, j * hstep, side)
            domain_lists[mean_class][variance_class].append((i * vstep, j * hstep, rotation, orientation))

    for mc in range(NUM_MEAN_CLASSES):
        for vc in range(NUM_VARIANCE_CLASSES):
            if len(domain_lists[mc][vc]) == 0:
                domain_lists[mc][vc] = [(0, 0, 0, 0)]
            domain_lists[mc][vc] = np.array(domain_lists[mc][vc], dtype=np.int32)

    return domain_lists

@njit
def _find_min_err_domain(img: np.ndarray, start_i: int, start_j: int, side: int,
         min_scale: np.float32, max_scale: np.float32, range_rot: int, range_orient: int,
         candidate_domains: np.ndarray) -> tuple[int, int, int, int, np.float32, np.float32, np.float32]:
    best_error = np.inf
    best_domain = (0, 0, 0, 0, 0., 0.)
    img_range = img[start_i:start_i + side, start_j:start_j + side].astype(np.float32)
    range_sum = np.sum(img_range)
    squared_range_sum = np.sum(img_range**2)

    for i in range(candidate_domains.shape[0]):
        domain_start_i, domain_start_j, domain_rot, domain_orient = candidate_domains[i][0], candidate_domains[i][1], candidate_domains[i][2], candidate_domains[i][3]
        # class is same if range is rotated by range_rot and oriented by range_orient and if
        # domain is rotated by domain_rot and oriented by domain_orient,
        # we need to find rotation/orientation of just domain that will give the same effect without rotating/orienting range
        if domain_orient == range_orient:
            d_rot = (domain_rot - range_rot) % 4
            d_orient = NORMAL_ORIENTATION
        else:
            d_rot = (domain_rot - range_rot + 2) % 4
            d_orient = TRANSPOSED_ORIENTATION

        domain_reordered = np.rot90(img[domain_start_i: domain_start_i + side * 2, domain_start_j: domain_start_j + side * 2], d_rot)
        if d_orient == TRANSPOSED_ORIENTATION:
            domain_reordered = domain_reordered.T

        domain_subsampled = average_subsample_jit(domain_reordered)
        error, scale_factor, offset_factor = _calc_rms_error(
            domain_subsampled, img_range, min_scale, max_scale, range_sum, np.sum(domain_subsampled),
            squared_range_sum, np.sum(domain_subsampled*domain_subsampled), img_range.shape[0]
        )

        if error < best_error:
            best_error = error
            best_domain = (domain_start_i, domain_start_j, d_orient, d_rot, scale_factor, offset_factor)

    return best_domain, best_error

@njit
def _calc_rms_error(domain: np.ndarray, range_: np.ndarray, min_scale: np.float32, max_scale: np.float32,
        range_sum: np.float32, domain_sum: np.float32, squared_range_sum: np.float32,
        squared_domain_sum: np.float32, range_size: int) -> tuple[np.float32, np.float32, np.float32]:
    n = range_size * range_size
    ab_sum = np.sum(domain * range_)
    
    denominator = (n * squared_domain_sum - domain_sum * domain_sum)
    if np.abs(denominator) < ZERO_TOLERANCE:
        s = 0.
        o = 1. / n * range_sum
    else:
        s = (n * ab_sum - range_sum * domain_sum) / denominator
        o = 1. / n * (range_sum - s * domain_sum)

    if s < min_scale or s > max_scale:
        s = min_scale
        o_left = 1. / n * (range_sum - s * domain_sum)
        R_left = _calc_square_error(n, squared_range_sum, s, squared_domain_sum, ab_sum, o_left, domain_sum, range_sum)
        s = max_scale
        o = 1. / n * (range_sum - s * domain_sum)
        R_right = _calc_square_error(n, squared_range_sum, s, squared_domain_sum, ab_sum, o, domain_sum, range_sum)
        if R_right < R_left:
            return R_right, s, o
        return R_left, min_scale, o_left
    else:
        R = _calc_square_error(n, squared_range_sum, s, squared_domain_sum, ab_sum, o, domain_sum, range_sum)
        return R, s, o

@njit
def _calc_square_error(n, squared_range_sum, s, squared_domain_sum, ab_sum, o, domain_sum, range_sum):
    # TODO use quantized scale and offset
    return 1. / n * (squared_range_sum + s * (s * squared_domain_sum - 2 * ab_sum + 2 * o * domain_sum) + o * (n * o - 2 * range_sum))

if __name__ == "__main__":
    # img = np.array([
    #     [0, 0, 4, 1],
    #     [0, 0, 4, 1],
    #     [2, 2, 4, 3],
    #     [2, 2, 4, 3],
    # ], dtype=np.float32)
    # print(_classify(img, 0, 0, 4))
    results = []
    for i in range(10000):
        img = np.random.random((8, 8))
        m, v, r, o = _classify(img, 0, 0, 8)
        results.append(v)
    from matplotlib import pyplot as plt
    print(np.mean(results))
    # should have uniform distribution
    plt.hist(results, bins=24)
    plt.show()
