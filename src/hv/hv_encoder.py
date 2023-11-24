import numpy as np

from model import Domain, EncodedImage
from utils import average_subsample


class HVEncoder:
    def __init__(self, img_raw: np.ndarray, domain_size=16, strides=(1, 1), threshold=1) -> None:
        """
        Args:
            img_raw (np.ndarray): 2d square array with side divisible by 16
        """
        self.img = img_raw
        self.domain_size = domain_size
        self.strides = strides
        self.sqr_diff_threshold = threshold

    def encode(self) -> EncodedImage:
        res = EncodedImage(self.img.shape[1], self.img.shape[0], range_size=0)
        res.domains = self.recursive_partitioning(self.img)
        print(f"-----------\nDOMAINS\n{res.domains}")
        return res
    
    def recursive_partitioning(self, img):
        print(f"REC_PART img s={len(img)}x{len(img[0])}")
        img_domain = self.domain_search(img)
        if img_domain is not None:
            return [img_domain]
        # no domain => let's partition
        P1, P2 = self.partition(img)
        print(f"partitions = sP1={len(P1)}x{len(P2[0])} ; sP2={len(P2)}x{len(P2[0])}")
        part1 = self.recursive_partitioning(P1)
        part2 = self.recursive_partitioning(P2)
        return [*part1, *part2]  
    
    def partition(self, image):
        N, M = len(image), len(image[0])
        sum_row = [0 for i in range(N)]
        sum_column = [0 for i in range(M)]
        for i in range(N):
            for j in range(M):
                sum_row[i] += image[i][j]
                sum_column[j] += image[i][j]
        #  biased differences
        v, h = [0 for i in range(N-1)], [0 for i in range(M-1)]
        for i in range(N-1):
            for j in range(M-1):
                v[i] = (min(N-j-1, j)/N-1)*(sum_row[i] - sum_row[i+1])
                h[j] = (min(M-j-1, j)/M-1)*(sum_column[j] - sum_column[j+1])
        h_max, v_max = np.argmax(h), np.argmax(v)
        if h_max > v_max:
            P1, P2 = np.split(np.array(image), [h_max], axis=0)
        else:
            P1, P2 = np.split(np.array(image), [v_max], axis=1)
        return P1.tolist(), P2.tolist()
    
    def domain_search(self, img):
        found_domain = None
        domain_diff = np.inf
        for i in range(0, self.img.shape[0], self.domain_size):
            for j in range(0, self.img.shape[1], self.domain_size):
                candidate_domain = self.img[i:i + self.domain_size,j:j + self.domain_size]
                x = self.sqr_diff(candidate_domain, img)
                if domain_diff is None or x < domain_diff:
                    # found better domain
                    domain_diff = x
                    found_domain = Domain(i, j, 0, 0, 0, 0)
        if domain_diff < self.sqr_diff_threshold:
            return found_domain  # found domain
        return None
    
    def sqr_diff(self, domain, img):
        averaged = average_subsample(domain)  # average 2x2
        avg_sum, img_sum = np.sum(averaged), np.sum(img)
        # x = (avg_sum - img_sum) * (avg_sum + img_sum)
        x = (avg_sum - img_sum)
        return x*x
    

