import numpy as np
from numba import int32, njit

from bruteforce.model import Domain, EncodedImage
from utils import average_subsample_jit


class JitedBruteForceEncoder:
    def __init__(self, img_raw: np.ndarray, range_size: int = 8, domain_size: int = 16) -> None:
        """
        Args:
            img_raw (np.ndarray): 2d square array with side divisible by 16
        """
        self.img = img_raw
        self.range_size = range_size
        self.domain_size = domain_size

    def encode(self) -> tuple[EncodedImage, list[np.float32]]:
        res = EncodedImage(self.img.shape[1], self.img.shape[0], self.range_size)
        errs = []

        for i in range(0, self.img.shape[0], self.range_size):
            for j in range(0, self.img.shape[1], self.range_size):
                domain_i, domain_j, orient, rot, s, o, R = _find_best_domain(
                    self.img, self.img[i:i + self.range_size,j:j + self.range_size], self.domain_size)
                
                res.domains.append(Domain(domain_i, domain_j, orient, rot, s, o))
                errs.append(np.sqrt(R))
        
        return res, errs

@njit
def _find_best_domain(img: np.ndarray, img_range: np.ndarray, domain_size: int32):
    # just check all possible domains for a given range

    best_domain = (0, 0, 0, 0, 0, 0, 0)
    best_rms = np.inf
    range_sum = np.sum(img_range)
    squared_range_sum = np.sum(img_range**2)

    for i in range(0, img.shape[0] - domain_size):
        for j in range(0, img.shape[1] - domain_size):
            domain = img[i:i+domain_size, j:j+domain_size].astype(np.float32)
            domains = [domain, domain.T]

            for orientation in range(2):
                oriented_domain = domains[orientation]
                for rotation in range(4):
                    rotated_domain = np.rot90(oriented_domain, rotation)
                    domain_subsampled = average_subsample_jit(rotated_domain)
                    
                    R, s, o = _calc_rms_error(
                        domain_subsampled, img_range, range_sum, np.sum(domain_subsampled),
                        squared_range_sum, np.sum(domain_subsampled*domain_subsampled), img_range.shape[0]
                    )

                    if R < best_rms:
                        best_rms = R
                        best_domain = (i, j, orientation, rotation, s, o, best_rms)

    return best_domain 


@njit
def _calc_rms_error(domain: np.ndarray, range_: np.ndarray,
                    range_sum: np.float32, domain_sum: np.float32, squared_range_sum: np.float32,
                    squared_domain_sum: np.float32, range_size: int32) -> tuple[np.float32, np.float32, np.float32]:
    n = range_size * range_size
    ab_sum = np.sum(domain * range_)
    
    denominator = (n * squared_domain_sum - domain_sum * domain_sum)
    if np.abs(denominator) < 1e-4:
        s = 0.
        o = 1. / n * range_sum
    else:
        s = (n * ab_sum - range_sum * domain_sum) / denominator
        o = 1. / n * (range_sum - s * domain_sum)

    if s < -1 or s > 1:
        s = -1.
        o_left = 1. / n * (range_sum - s * domain_sum)
        R_left = _calc_R_error(n, squared_range_sum, s, squared_domain_sum, ab_sum, o_left, domain_sum, range_sum)
        s = 1
        o = 1. / n * (range_sum - s * domain_sum)
        R_right = _calc_R_error(n, squared_range_sum, s, squared_domain_sum, ab_sum, o, domain_sum, range_sum)
        if R_right < R_left:
            return R_right, s, o
        return R_left, -1., o_left
    else:
        R = _calc_R_error(n, squared_range_sum, s, squared_domain_sum, ab_sum, o, domain_sum, range_sum)
        return R, s, o

@njit
def _calc_R_error(n, squared_range_sum, s, squared_domain_sum, ab_sum, o, domain_sum, range_sum):
    return  1. / n * (squared_range_sum + s * (s * squared_domain_sum - 2 * ab_sum + 2 * o * domain_sum) + o * (n * o - 2 * range_sum))