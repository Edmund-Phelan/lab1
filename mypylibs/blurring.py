import numpy as np

from scipy.ndimage import convolve


# Builds the 2D Gaussian weights used to average each pixel with its neighbours.
def gaussian_kernel_2d(sigma: float) -> np.ndarray:
    radius = int(np.ceil(3 * sigma))  # 3 sigma covers about 99.7% of the curve
    x = np.arange(-radius, radius + 1)
    xx, yy = np.meshgrid(x, x)
    kernel = np.exp(-(xx**2 + yy**2) / (2 * sigma**2))
    return kernel / kernel.sum()  # weights sum to 1 to keep brightness the same


# Blurs the 2D greyscale image
def gaussian_blur(img: np.ndarray, sigma: float) -> np.ndarray:
    kernel = gaussian_kernel_2d(sigma)
    # Making it a float keeps the decimals
    return convolve(img.astype(float), kernel, mode='nearest')
