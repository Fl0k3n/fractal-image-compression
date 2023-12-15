import numpy as np

from model import Domain, EncodedImage
from utils import average_subsample


class HVDecoder:
    def __init__(self, domain_size=32, min_range_size=4):
        self.domain_size = domain_size
        self.min_range_size = min_range_size

    # def decode(self, encoded_img: EncodedImage, iterations=32) -> np.ndarray:
    #     img = np.zeros((encoded_img.height, encoded_img.width), dtype=np.float32) 
    #     next_img = np.empty_like(img)
    #     for _ in range(iterations):
    #         for range_idx, dom in enumerate(encoded_img.domains):

