
import yaml
import numpy as np

from PIL import Image
from os import path as opath
from mypylibs import trajectory
import matplotlib.pyplot as plt
from mypylibs.greyscale import get_greyscale

ROOT_IN = 'in'
ROOT_OUT = 'out'
EXT = 'png'

def main(case: int):

    img = Image.open(opath.join(ROOT_IN, f'{case:02}.{EXT}'))
    imgArray = np.asarray(img.convert('RGB'))
    grey_img = get_greyscale(imgArray)
    print(grey_img.shape)
    
    with open(opath.join(ROOT_IN, f'{case:02}.yaml')) as f:
        params = yaml.safe_load(f)

    # Expected outputs per case
    # plt.savefig(opath.join(ROOT_OUT, f'{case:02}_transforms.{EXT}'))
    # plt.savefig(opath.join(ROOT_OUT, f'{case:02}_trajectory.{EXT}'))
    # plt.savefig(opath.join(ROOT_OUT, f'{case:02}_final.{EXT}'))

if __name__ == '__main__':

    case = 0
    while(True):
        case += 1
        try:
            main(case)
        except FileNotFoundError:
            break
