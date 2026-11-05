
import yaml
import numpy as np

from PIL import Image
from os import path as opath
from mypylibs import trajectory
import matplotlib.pyplot as plt
from mypylibs.greyscale import get_greyscale
from mypylibs.blurring import gaussian_blur
from mypylibs.thresholding import apply_threshold
from mypylibs.borders import get_borders

ROOT_IN = 'in'
ROOT_OUT = 'out'
EXT = 'png'

def main(case: int):

    img = Image.open(opath.join(ROOT_IN, f'{case:02}.{EXT}'))
    imgArray = np.asarray(img.convert('RGB'))
    
    with open(opath.join(ROOT_IN, f'{case:02}.yaml')) as f:
        params = yaml.safe_load(f)

    grey_img = get_greyscale(imgArray)
    blurred_img = gaussian_blur(grey_img, sigma=params['blur']['sigma'])
    thresholded_img = apply_threshold(blurred_img, threshold=params['threshold']['value'])
    border_img = get_borders(
        thresholded_img,
        threshold=params['borders']['relative_threshold'],
    )
    print(border_img.shape)

    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    panels = [
        (imgArray, 'Original image', None),
        (grey_img, 'Greyscale', 'gray'),
        (blurred_img, 'Gaussian blur', 'gray'),
        (thresholded_img, 'Threshold', 'gray'),
        (border_img, 'Detected borders', 'gray'),
    ]

    for ax, (image, title, cmap) in zip(axes.flat, panels):
        ax.imshow(image, cmap=cmap, vmin=0, vmax=255)
        ax.set_title(title)
        ax.axis('off')

    axes.flat[-1].axis('off')
    fig.tight_layout()
    fig.savefig(opath.join(ROOT_OUT, f'{case:02}_transforms.{EXT}'))
    plt.close(fig)

if __name__ == '__main__':

    case = 0
    while(True):
        case += 1
        print("case", case)
        try:
            main(case)
        except FileNotFoundError:
            print("File not found for case", case)
            break
