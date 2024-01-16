from hvv2.common import HVNode
from hvv2.encoder import HVEncoder
from hvv2.decoder import HVDecoder
from hvv2.serialization import HVDeserializer, HVSerializer
import utils
from time import time
import numpy as np


def load_img(path: str) -> np.ndarray:
    return utils.load_grayscale(path)

def kalafior(size: int = 64) -> np.ndarray:
    path = "imgs\\raw\\cauliflower_64x64_colored.png"
    # return load_img(f'../imgs/raw/cauliflower_{size}x{size}_colored.png')
    return load_img(path)


img = kalafior()
encoder = HVEncoder(4, 4, 32, 32, 7, 5, 4., 1.)
start = time()
encoded_img = encoder.encode(img)
encoding_time = time() - start

start = time()
decoding_iterations=2
decoded_img = HVDecoder(decoding_iterations, stop_on_relative_error=5e-4, log_stop=True).decode(encoded_img)
decoding_time = time() - start

print(f"Encoding time: {encoding_time}; Decoding time: {decoding_time}")
