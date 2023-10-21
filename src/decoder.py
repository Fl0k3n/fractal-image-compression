import numpy as np

from model import EncodedImage
from utils import average_subsample


class Decoder:
    def __init__(self) -> None:
        self.range_size = 8
        self.domain_size = 16

    def decode(self, encoded_img: EncodedImage, iterations=32) -> np.ndarray:
        # img = np.random.random_integers(0, 255, (encoded_img.height, encoded_img.width))
        img = np.zeros((encoded_img.height, encoded_img.width), dtype=np.float32) 
        next_img = np.empty_like(img)

        for _ in range(iterations):
            for range_idx, dom in enumerate(encoded_img.domains):
                range_i = (range_idx // (encoded_img.width // self.range_size)) * self.range_size
                range_j = (range_idx % (encoded_img.height // self.range_size)) * self.range_size

                dom_i = np.copy(img[dom.start_i:dom.start_i + self.domain_size, dom.start_j:dom.start_j + self.domain_size])
                if dom.orientation == 1:
                    dom_i = dom_i.T
                dom_i = np.rot90(dom_i, dom.rotation)
                dom_i = average_subsample(dom_i) * dom.s + dom.o

                next_img[range_i:range_i + self.range_size, range_j:range_j + self.range_size] = dom_i

            img = next_img
        
        return img.astype(np.uint8)
        