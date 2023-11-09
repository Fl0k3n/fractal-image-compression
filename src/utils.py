import numpy as np
from numba import njit
from PIL import Image


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
    dom = arr.reshape(-1).reshape((-1, arr.shape[1] * 2))
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
    