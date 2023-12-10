from typing import Literal

import numpy as np
from numba import njit
from PIL import Image

ChromaSubsampling = Literal["4:4:4", "4:2:0"]

def load_grayscale(src_path: str) -> np.ndarray:
    with Image.open(src_path) as im:
        return np.array(im.convert("L"))

def average_subsample(arr: np.ndarray) -> np.ndarray:
    """ Divides arr into 2x2 squares and averages them

    Args:
        arr (np.ndarray): 2d square matrix, dtype should fit 4*255

    Returns:
        np.ndarray: subsampled 2d square matrix, dtype float32
    """
    dom = arr.reshape(-1).reshape((-1, arr.shape[1] * 2)).astype(np.int32)
    x = dom[:,:arr.shape[1]] + dom[:,arr.shape[1]:]
    x = x[:,::2] + x[:,1::2]
    return x / 4

@njit
def average_subsample_jit(arr: np.ndarray) -> np.ndarray:
    res = np.empty((arr.shape[0] // 2, arr.shape[1] // 2), dtype=np.float32)
    for i in range(0, arr.shape[0], 2):
        for j in range(0, arr.shape[1], 2):
            s = np.float32(arr[i, j]) + np.float32(arr[i, j + 1]) + np.float32(arr[i + 1, j]) + np.float32(arr[i + 1, j + 1])
            res[i // 2, j // 2] = s / 4.
    return res

def rgb_to_ycbcr(rgb_img: np.ndarray, chroma_subsampling: ChromaSubsampling) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    N, M, _ = rgb_img.shape
    # ycbcr_img = np.array(Image.fromarray(rgb_img).convert('YCbCr').split())
    x = rgb_img.astype(np.float64) / 256.
    R, G, B = x[...,0], x[...,1], x[...,2]
    Y = 16. + 65.738 * R + 129.057 * G + 25.064 * B
    Cb = 128. - 37.945 * R - 74.494 * G + 112.439 * B
    Cr = 128. + 112.439 * R - 94.154 * G - 18.285 * B
    ycbcr_img = np.stack((Y, Cb, Cr), axis=-1).clip(0., 255.).astype(np.uint8)

    if chroma_subsampling == "4:4:4":
        return ycbcr_img[...,0], ycbcr_img[...,1], ycbcr_img[...,2]
    elif chroma_subsampling == "4:2:0":
        cb_subs = np.empty((N // 2 + N % 2, M // 2 + M % 2), dtype=np.uint8)
        cr_subs = np.empty((N // 2 + N % 2, M // 2 + M % 2), dtype=np.uint8)
        for i, band in enumerate((cb_subs, cr_subs)):
            band[:band.shape[0] - N % 2, :band.shape[1] - M % 2] = average_subsample(ycbcr_img[:N - N % 2, :M - M % 2, i + 1])
            if M % 2 == 1:
                band[:band.shape[0]-N%2, -1] = ((
                    ycbcr_img[:N-N%2:2, -1, i+1].astype(np.int32) + ycbcr_img[1:N-N%2:2, -1, i+1].astype(np.int32)
                ) / 2).astype(np.uint8)
            if N % 2 == 1:
                band[-1 :band.shape[1]-M%2] = ((
                    ycbcr_img[-1, :M-M%2:2, i+1].astype(np.int32) + ycbcr_img[-1, 1:M-M%2:2, i+1].astype(np.int32)
                ) / 2).astype(np.uint8)
            if M % 2 == 1 and N % 2 == 1:
                band[-1, -1] = ycbcr_img[-1, -1, i+1]
        return ycbcr_img[...,0], cb_subs, cr_subs
    raise Exception("Unsupported chroma subsampling")

def ycbcr_to_rgb(ycbcr_img: tuple[np.ndarray, np.ndarray, np.ndarray], chroma_subsampling: ChromaSubsampling) -> np.ndarray:
    if chroma_subsampling == "4:4:4":
        ycbcr_upsampled = ycbcr_img
    elif chroma_subsampling == "4:2:0":
        N, M = ycbcr_img[0].shape
        Ns, Ms = ycbcr_img[1].shape
        cb_ups = np.empty_like(ycbcr_img[0])
        cr_ups = np.empty_like(ycbcr_img[0])
        for i, band in enumerate((cb_ups, cr_ups)):
            band[:N-N%2:2,:M-M%2:2]   = ycbcr_img[i+1][:Ns-Ns%2,:Ms-M%2]
            band[1:N-N%2:2,:M-M%2:2]  = ycbcr_img[i+1][:Ns-Ns%2,:Ms-M%2]
            band[:N-N%2:2,1:M-M%2:2]  = ycbcr_img[i+1][:Ns-Ns%2,:Ms-M%2]
            band[1:N-N%2:2,1:M-M%2:2] = ycbcr_img[i+1][:Ns-Ns%2,:Ms-M%2]
            if M % 2 == 1:
                band[:N-N%2:2, -1]  = ycbcr_img[i+1][:, -1]
                band[1:N-N%2:2, -1] = ycbcr_img[i+1][:, -1]
            if N % 2 == 1:
                band[-1, :M-M%2:2]  = ycbcr_img[i+1][-1, :]
                band[-1, 1:M-M%2:2] = ycbcr_img[i+1][:, -1]
            if M % 2 == 1 and N % 2 == 1:
                band[-1, -1] = ycbcr_img[i+1][-1, -1]
        ycbcr_upsampled = (ycbcr_img[0], cb_ups, cr_ups)
    else:
        raise Exception("Unsupported chroma subsampling")

    # ycbcr_upsampled = [Image.fromarray(dim) for dim in ycbcr_upsampled]
    # return np.array(Image.merge("YCbCr", ycbcr_upsampled).convert('RGB'))
    Y, Cb, Cr = np.array(ycbcr_upsampled, dtype=np.float64) / 256.
    R = 298.082 * Y + 408.583 * Cr - 222.921
    G = 298.082 * Y - 100.291 * Cb - 208.120 * Cr + 135.576
    B = 298.082 * Y + 516.412 * Cb - 276.836
    return np.stack((R, G, B), axis=-1).clip(0., 255.).astype(np.uint8)

if __name__ == "__main__":
    arr = np.arange(4 * 7 * 3).reshape((4, 7, 3)).astype(np.uint8)
    my_ycbcr = rgb_to_ycbcr(arr, "4:2:0")
    my_rgb = ycbcr_to_rgb(my_ycbcr, "4:2:0")
    print(arr)
    print()
    print(my_rgb)