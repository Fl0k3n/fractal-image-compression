from time import time
from PIL import Image

from hv.hv_encoder import *
from decoder import Decoder

if __name__ == "__main__":
    src_path = ".\\imgs\\raw\\cauliflower_128x128_colored.png"
    img=None
    with Image.open(src_path) as im:
        img =  np.array(im.convert("L"))
    assert img is not None
    start = time()
    encoder = HVEncoder(img)
    encoded_img = encoder.encode()
    encoding_time = time() - start
    print(f"Encoding time = {encoding_time}; domains_count = {len(encoded_img.domains)}")

    # start = time()
    # decoder = Decoder()
    # decoded_img = decoder.decode(encoded_img, iterations=32)
    # decoding_time = time() - start
    # print(f"Decoding time = {encoding_time}")

    

    # TODO:
    # kodowanie entropijne
    # jpeg 2000
