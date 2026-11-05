
import yaml
import numpy as np

from PIL import Image
from os import path as opath
from mypylibs import trajectory
from mypylibs.greyscale import get_greyscale
from mypylibs.blurring import gaussian_blur
from mypylibs.thresholding import apply_threshold
from mypylibs.borders import get_borders
from mypylibs import fourier
from mypylibs.visualization import save_transform, save_trajectory, save_final, save_animation

ROOT_IN = 'in'
ROOT_OUT = 'out'

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

    save_transform(
        imgArray,
        grey_img,
        blurred_img,
        thresholded_img,
        border_img,
        opath.join(ROOT_OUT, f'{case:02}_transforms.png'),
    )

    # Silhouette trajectory: ordered, uniformly resampled boundary points
    x, y = trajectory.calculate(border_img, num_samples=500)
    save_trajectory(x, y, border_img.shape, opath.join(ROOT_OUT, f'{case:02}_trajectory.png'))

    # Fourier epicycles: reconstruct with 5%..100% of the dominant harmonics
    Z, freqs = fourier.descriptors(x, y)
    N = len(Z)
    percents = range(5, 101, 5)
    recons = [fourier.reconstruct(Z, freqs, max(1, round(N * p / 100)), N) for p in percents]
    save_final(recons, percents, border_img.shape, opath.join(ROOT_OUT, f'{case:02}_final.png'))

    # Bonus: MP4 of the rotating epicycles
    save_animation(Z, freqs, border_img.shape, opath.join(ROOT_OUT, f'{case:02}.mp4'))

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
