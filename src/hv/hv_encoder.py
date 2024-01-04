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

    def get_size(self, P):
        return min([P.shape[0], P.shape[1]]) 
        
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
                Queue.append(P1)
                Queue.append(P2)
                # Queue.sort(key=self.compare)
                Queue = sorted(Queue, key=self.get_size, reverse=True)
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
        return P1, P2
    
    def domain_search(self, img_range):
        domain_idx = 0
        # found_domain = None
        found_partition = None
        found_range = None
        domain_diff = np.inf
        domain_shape = (2*img_range.shape[0], 2*img_range.shape[1])
        if domain_shape[0] > self.domain_size or domain_shape[1] > self.domain_size:
            return None
        for i in range(0, self.img.shape[0] - domain_shape[0], domain_shape[0]//2):
            for j in range(0, self.img.shape[1] - domain_shape[1], domain_shape[1]//2):
                candidate_domain = self.img[i:i + domain_shape[0],j:j + domain_shape[1]]

                # for orientation, oriented_domain in enumerate((candidate_domain, candidate_domain.T)):
                #     for rotation in range(4):
                orientation = 0
                oriented_domain = candidate_domain
                rotation = 0
                #
                rotated_domain = np.rot90(oriented_domain, rotation)
                domain_subsampled = average_subsample(rotated_domain)
                x = self.sqr_diff(domain_subsampled, img_range)
                if domain_diff is None or x < domain_diff:
                    # found better domain
                    domain_diff = x
                    s, o = self.getSO(domain_subsampled, img_range)
                    # found_domain = Domain(i, j, rotation, s, o)
                    found_partition = HvPartition(i, j, domain_idx, orientation, rotation, s, o)
                    found_range = HvRange(i,  j, img_range.shape[0], img_range.shape[1], domain_idx, orientation, rotation, s, o)
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
        ab_sum = np.sum(domain_subsampled * img_range)
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
        


