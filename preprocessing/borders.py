import numpy as np
from scipy.ndimage import convolve


def get_borders(image, threshold=0.5):
    """
    Detect object borders in a greyscale/binary uint8 image (0-255).

    Applies hand-built Sobel derivative kernels, combines the horizontal and
    vertical gradients into an edge-strength magnitude, and thresholds it.

    Returns:
    numpy.ndarray: uint8 image (0-255) where border pixels are 255, else 0.
    """
    sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])

    image = image.astype(float)
    gx = convolve(image, sobel_x, mode="nearest")
    gy = convolve(image, sobel_y, mode="nearest")

    magnitude = np.sqrt(gx ** 2 + gy ** 2)
    borders = magnitude > (threshold * magnitude.max())

    return (borders * 255).astype(np.uint8)


