import numpy as np

from model import Domain, EncodedImage
from utils import average_subsample
from hv.common import *


class HVEncoder:
    def __init__(self, img_raw: np.ndarray, domain_size=32, strides=(16, 16), threshold=1000000.0) -> None:
        """
        Args:
            img_raw (np.ndarray): 2d square array with side divisible by 16
        """
        self.img = img_raw
        self.domain_size = domain_size
        self.strides = strides
        self.sqr_diff_threshold = threshold
        self.min_range_size = 4

    def encode(self) -> EncodedHvImage:
        res = EncodedHvImage(self.img.shape[1], self.img.shape[0])
        res.domains = self.queue_partitioning(self.img)
        return res  
    
    def queue_partitioning(self, first_img):
        Queue = [first_img]
        partitions = []
        while len(Queue) > 0:
            img = Queue.pop()
            img_domain = self.domain_search(img)
            if img_domain is not None:
                partitions.append(img_domain)
            else:
                P1, P2 = self.partition(img)
                p1_x, p1_y = P1.shape
                p2_x, p2_y = P2.shape
                p_sizes = [P1.shape[0], P2.shape[0], P1.shape[1], P2.shape[1]]
                p_max = np.argmin(p_sizes)
                if p_max % 2 == 0:
                    Queue.append(P2)
                    Queue.append(P1)
                else:
                    Queue.append(P1)
                    Queue.append(P2)
        return partitions
    
    def partition(self, image):
        N, M = len(image), len(image[0])
        sum_row = [0 for _ in range(N)]
        sum_column = [0 for _ in range(M)]
        for i in range(N):
            for j in range(M):
                sum_row[i] += image[i][j]
                sum_column[j] += image[i][j]
        v = [(min(N-i-1, i)/(N-1))*(sum_row[i] - sum_row[i+1]) for i in range(N-1)]
        h = [(min(M-j-1, j)/(M-1))*(sum_column[j] - sum_column[j+1]) for j in range(M-1)]
        if len(v) < self.min_range_size:
            h_max = np.argmax(h)
            P1, P2 = np.split(np.array(image), [h_max+1], axis=1)
        elif len(h) < self.min_range_size:
            v_max = np.argmax(v)
            P1, P2 = np.split(np.array(image), [v_max+1], axis=0)
        else:
            h_max, v_max = np.argmax(h), np.argmax(v)
            if h[h_max] > v[v_max]:
                P1, P2 = np.split(np.array(image), [h_max+1], axis=1)
            else:
                P1, P2 = np.split(np.array(image), [v_max+1], axis=0)
        assert 0 not in P1.shape
        assert 0 not in P1.shape
        return P1, P2
    
    def domain_search(self, img_range):
        domain_idx = 0
        found_domain = None
        domain_diff = np.inf
        if 2 * max(img_range.shape) > self.domain_size:
            return None
        for i in range(0, self.img.shape[0] - self.domain_size, self.strides[0]):
            for j in range(0, self.img.shape[1] - self.domain_size, self.strides[1]):
                candidate_domain = self.img[i:i + self.domain_size,j:j + self.domain_size]

                for orientation, oriented_domain in enumerate((candidate_domain, candidate_domain.T)):
                    for rotation in range(4):
                        rotated_domain = np.rot90(oriented_domain, rotation)
                        domain_subsampled = average_subsample(rotated_domain)
                        x = self.sqr_diff(domain_subsampled, img_range)
                        if domain_diff is None or x < domain_diff:
                            # found better domain
                            domain_diff = x
                            s, o = self.getSO(domain_subsampled, img_range)
                            # found_domain = Domain(i, j, rotation, s, o)
                            found_partition = HvPartition(
                                i, 
                                j,
                                domain_idx,
                                orientation, 
                                rotation, 
                                s, 
                                o
                            )
                            found_range = HvRange(
                                i, 
                                j, 
                                img_range.shape[0], 
                                img_range.shape[1], 
                                domain_idx, 
                                orientation, 
                                rotation, 
                                s, 
                                o
                            )
                            # found_domain = HvDomain(i, j, img_range.shape[0], img_range.shape[1], orientation, rotation, s, o)
                domain_idx += 1
        p, q = img_range.shape
        if domain_diff <= self.sqr_diff_threshold or p <= self.min_range_size or q <= self.min_range_size:
            # return found_domain
            return found_partition, found_range
        return None
    
    def sqr_diff(self, domain, img_range):
        pixel_diffs = 0
        scale = (domain.shape[0] // img_range.shape[0], domain.shape[1] // img_range.shape[1])
        for i in range(img_range.shape[0]):
            for j in range(img_range.shape[1]):
                dom_pixel = domain[i*scale[0], j*scale[1]]
                diff = dom_pixel - img_range[i, j]
                pixel_diffs += diff * diff
        return pixel_diffs
    
    def getSO(self, img_range, domain_subsampled):
        abs_product = [
            [domain_subsampled[i][j] * img_range[i][j] 
             for j in range(min(domain_subsampled.shape[1], img_range.shape[1]))
             ]
        for i in range(min(domain_subsampled.shape[0], img_range.shape[0]))
        ]
        ab_sum = np.sum(abs_product)
        domain_sum = np.sum(domain_subsampled)
        squared_domain_sum = np.sum(domain_subsampled**2)
        range_sum = np.sum(img_range)
        range_size = min(img_range.shape[0], img_range.shape[1])
        n =  range_size * range_size
        denominator = (n * squared_domain_sum - domain_sum * domain_sum)
        if np.abs(denominator) < 1e-4:
            s = 0.
            o = 1. / n * range_sum
        else:
            s = (n * ab_sum - range_sum * domain_sum) / denominator
            o = 1. / n * (range_sum - s * domain_sum)
        return s, o
        


