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
    dom = np.copy(arr).reshape(-1).reshape((-1, arr.shape[1] * 2))
    x = dom[:,:arr.shape[1]] + dom[:,arr.shape[1]:]
    x = x[:,::2] + x[:,1::2]
    return x / 4
    