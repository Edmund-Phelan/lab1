from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


def get_greyscale(image):
    """
    Convert an RGB uint8 image (0-255) to greyscale uint8 (0-255).

    Returns:
    numpy.ndarray: Greyscale image.
    """
    # Check if the input image is already in greyscale
    if len(image.shape) == 2:
        return image

    # BT.601 luma weights
    grey_image = 0.299 * image[:, :, 0] + 0.587 * image[:, :, 1] + 0.114 * image[:, :, 2]

    return np.rint(grey_image).astype(np.uint8)

