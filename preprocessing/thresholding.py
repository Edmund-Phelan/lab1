
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from pathlib import Path


def apply_threshold(image: np.ndarray, threshold: int = 127) -> np.ndarray:
    if image.ndim != 2:
        raise ValueError("Thresholding requires a 2D grayscale image.")

    binary_image = np.where(image >= threshold, 255, 0)

    return binary_image.astype(np.uint8)


#TESTING
#Load image as a grayscale NumPy array
project_root = Path(__file__).resolve().parent.parent
image_path = project_root / "in" / "01.jpg"
image = np.array(Image.open(image_path).convert("L"))

#Apply thresholding
thresholded_image = apply_threshold(image, 240)

#Display both images
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.imshow(image, cmap="gray", vmin=0, vmax=255)
plt.title("Original Grayscale")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(thresholded_image, cmap="gray", vmin=0, vmax=255)
plt.title("Thresholded Image")
plt.axis("off")

plt.tight_layout()
plt.show()
