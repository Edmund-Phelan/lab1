
import yaml
import numpy as np

from PIL import Image
from os import path as opath
from mypylibs import trajectory
from mypylibs.greyscale import get_greyscale
from mypylibs.blurring import gaussian_blur
from mypylibs.thresholding import apply_threshold
from mypylibs.borders import get_borders
from mypylibs.visualization import save_transform

ROOT_IN = 'in'
ROOT_OUT = 'out'
EXT = 'png'

def main(case: int):

    with open(opath.join(ROOT_IN, f'{case:02}.yaml')) as f:
            params = yaml.safe_load(f)

    img = Image.open(opath.join(ROOT_IN, f'{case:02}.{params['image']['extension']}'))
    imgArray = np.asarray(img.convert('RGB'))
    
    

    grey_img = get_greyscale(imgArray)
    blurred_img = gaussian_blur(grey_img, sigma=params['blur']['sigma'])
    thresholded_img = apply_threshold(blurred_img, threshold=params['threshold']['value'])
    border_img = get_borders(
        thresholded_img,
        threshold=params['borders']['relative_threshold'],
    )
    print(border_img.shape)

    save_transform(
        imgArray,
        grey_img,
        blurred_img,
        thresholded_img,
        border_img,
        opath.join(ROOT_OUT, f'{case:02}_transforms.{params["image"]["extension"]}'),
    )

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
