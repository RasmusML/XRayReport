import os
import numpy as np
from PIL import Image
import cv2


def count_files(path):
    return len([name for name in os.listdir(path) if os.path.isfile(os.path.join(path, name))])


def get_filenames(path):
    return [name for name in os.listdir(path) if os.path.isfile(os.path.join(path, name))]


def load_png(path):
    with open(path, "rb") as f:
        with Image.open(f) as img:
            return np.array(img)


def crop_center(img, cropXY):
    cropx, cropy = cropXY
    y,x, _ = img.shape
    startx = x // 2 - (cropx // 2)
    starty = y // 2 - (cropy // 2)
    return img[starty:starty+cropy, startx:startx+cropx]


def crop_and_scale(img, cropXY):
    y,x, _ = img.shape

    dim = min(x, y)
    img = crop_center(img, (dim, dim))
    
    return cv2.resize(img, cropXY)