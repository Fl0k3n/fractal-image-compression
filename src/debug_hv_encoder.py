from time import time
from PIL import Image

from hv.hv_encoder import *

    

    

if __name__ == "__main__":
    src_path = "fractal-image-compression\\imgs\\raw\cauliflower_128x128_colored.png"
    img=None
    with Image.open(src_path) as im:
        img =  np.array(im.convert("L"))
    assert img is not None
    start = time()
    encoder = HVEncoder(img)
    encoded_img = encoder.encode()
    encoding_time = time() - start
    print(f"Encoding time = {encoding_time}")

