
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
    grey_img = get_greyscale(imgArray)
    blurred_img = gaussian_blur(grey_img, sigma=1.0)
    thresholded_img = apply_threshold(blurred_img, threshold=0.5)
    border_img = get_borders(thresholded_img, threshold=0.5)
    print(border_img.shape)
    
    with open(opath.join(ROOT_IN, f'{case:02}.yaml')) as f:
        params = yaml.safe_load(f)

    # Expected outputs per case
    plt.savefig(opath.join(ROOT_OUT, f'{case:02}_transforms.{EXT}'))
    plt.savefig(opath.join(ROOT_OUT, f'{case:02}_trajectory.{EXT}'))
    plt.savefig(opath.join(ROOT_OUT, f'{case:02}_final.{EXT}'))

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
