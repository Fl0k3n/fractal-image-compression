import numpy as np

from bruteforce.model import Domain, EncodedImage
from utils import average_subsample


class BruteForceEncoder:
    def __init__(self, img_raw: np.ndarray, range_size=8, domain_size=16, strides=(1, 1)) -> None:
        """
        Args:
            img_raw (np.ndarray): 2d square array with side divisible by 16
        """
        self.img = img_raw
        self.range_size = range_size
        self.domain_size = domain_size
        self.strides = strides

    def encode(self) -> EncodedImage:
        res = EncodedImage(self.img.shape[1], self.img.shape[0], self.range_size)

        for i in range(0, self.img.shape[0], self.range_size):
            for j in range(0, self.img.shape[1], self.range_size):
                domain_i, domain_j, orient, rot, s, o = self._find_best_domain(
                    self.img[i:i + self.range_size,j:j + self.range_size])
                res.domains.append(Domain(domain_i, domain_j, orient, rot, s, o))
        
        return res

    def _find_best_domain(self, img_range: np.ndarray):
        best_domain = None
        best_rms = np.inf
        range_sum = np.sum(img_range)
        squared_range_sum = np.sum(img_range**2)

        for i in range(0, self.img.shape[0] - self.domain_size, self.strides[0]):
            for j in range(0, self.img.shape[1] - self.domain_size, self.strides[1]):
                domain = self.img[i:i+self.domain_size, j:j+self.domain_size].astype(np.float32)
    
                for orientation, oriented_domain in enumerate((domain, domain.T)):
                    for rotation in range(4):
                        rotated_domain = np.rot90(oriented_domain, rotation)
                        domain_subsampled = average_subsample(rotated_domain)
                        
                        R, s, o = self._calc_rms_error(
                            domain_subsampled, img_range, range_sum, np.sum(domain_subsampled),
                            squared_range_sum, np.sum(domain_subsampled**2)
                        )

                        if R < best_rms:
                            best_rms = R
                            best_domain = (i, j, orientation, rotation, s, o)

        return best_domain 
    
    # domain referes to "a", range referes to "b" from the book
    def _calc_rms_error(self, domain: np.ndarray, range_: np.ndarray,
                        range_sum: np.float32, domain_sum: np.float32, squared_range_sum: np.float32,
                        squared_domain_sum: np.float32) -> tuple[np.float32, np.float32, np.float32]:
        n =  self.range_size * self.range_size
        ab_sum = np.sum(domain * range_)
        
        denominator = (n * squared_domain_sum - domain_sum * domain_sum)
        if np.abs(denominator) < 1e-4:
            s = 0.
            o = 1. / n * range_sum
        else:
            s = (n * ab_sum - range_sum * domain_sum) / denominator
            o = 1. / n * (range_sum - s * domain_sum)
        R =  1. / n * (squared_range_sum + s * (s * squared_domain_sum - 2 * ab_sum + 2 * o * domain_sum) + o * (n * o - 2 * range_sum))
        # no reason to take sqrt
        return R, s, o
    

if __name__ == "__main__":
    # x = np.ones((16, 16))
    # x[2, 0] = 3
    # x[2, 1] = 4
    # x[3, 0] = 5
    # x[3, 1] = 4
    x = np.arange(64).reshape((16, 16))
    print(x)
    print(average_subsample(x))
