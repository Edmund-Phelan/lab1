from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def save_transform(
    original: np.ndarray,
    greyscale: np.ndarray,
    blurred: np.ndarray,
    thresholded: np.ndarray,
    borders: np.ndarray,
    output_path: str | Path,
) -> None:
    panels = [
        (original, 'Original image', None),
        (greyscale, 'Greyscale', 'gray'),
        (blurred, 'Gaussian blur', 'gray'),
        (thresholded, 'Threshold', 'gray'),
        (borders, 'Detected borders', 'gray'),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    for ax, (image, title, cmap) in zip(axes.flat, panels):
        ax.imshow(image, cmap=cmap, vmin=0, vmax=255)
        ax.set_title(title)
        ax.axis('off')

    axes.flat[-1].axis('off')
    fig.tight_layout()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)
