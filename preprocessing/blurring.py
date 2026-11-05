import numpy as np

from scipy.ndimage import convolve1d


def gaussian_kernel_1d(sigma: float) -> np.ndarray:
    radius = int(np.ceil(3 * sigma))
    x = np.arange(-radius, radius + 1)
    kernel = np.exp(-x**2 / (2 * sigma**2))
    return kernel / kernel.sum()


def gaussian_blur(img: np.ndarray, sigma: float) -> np.ndarray:
    kernel = gaussian_kernel_1d(sigma)
    blurred = convolve1d(img.astype(float), kernel, axis=0, mode='nearest')
    return convolve1d(blurred, kernel, axis=1, mode='nearest')
