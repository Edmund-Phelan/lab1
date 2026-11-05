
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from pathlib import Path


def apply_threshold(image: np.ndarray, threshold: int = 240) -> np.ndarray:
    if image.ndim != 2:
        raise ValueError("Thresholding requires a 2D grayscale image.")

    binary_image = np.where(image >= threshold, 255, 0)

    return binary_image.astype(np.uint8)
