import numpy as np

# from model import Domain, EncodedImage
from utils import average_subsample
from hv.common import *

class HVDecoder:
    def __init__(self, domain_size=32, min_range_size=4, strides=(16, 16)):
        self.domain_size = domain_size
        self.min_range_size = min_range_size
        self.strides = strides

    def decode(self, encoded_img: EncodedHvImage, iterations=32):
        ranges = encoded_img.ranges
        img = np.zeros((encoded_img.height, encoded_img.width), dtype=np.float32) 
        next_img = np.empty_like(img)

        for _ in range(iterations):
            for img_range in ranges:
                domains = self.pre_calculate((encoded_img.height, encoded_img.width), (2*img_range.len_i, 2*img_range.len_j))
                dom = domains[img_range.domain_idx]
                range_i, range_j = img_range.start_i, img_range.start_j
                size_i, size_j = img_range.len_i, img_range.len_j
                
                dom_i = np.copy(img[dom.start_i:dom.start_i + self.domain_size, dom.start_j:dom.start_j + self.domain_size])
                if dom.orientation == 1:
                    dom_i = dom_i.T
                dom_i = np.rot90(dom_i, dom.rotation)
                dom_i = average_subsample(dom_i) * dom.s + dom.o

                next_img[range_i:range_i + size_i, range_j:range_j + size_j] = dom_i
            img = next_img
        return img.astype(np.uint8)
    
    def pre_calculate(self, img_shape, domain_shape):
        '''
        TODO:
        Based on range information, assign a domain pixel to each of the range pixels. 
        This domain pixel is the first of four that have been subsampled/averaged in the encoding process.
        scaling and offset are stored only once per range
        '''
        domains = []
        for i in range(0, img_shape[0] - domain_shape[0], domain_shape[0]//2):
            for j in range(0, img_shape[1] - domain_shape[1], domain_shape[1]//2):
                domains.append(HvDomain(i, j))
        return domains


