import os
import numpy as np
from PIL import Image
import cv2
import torch
import pickle

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

def save_dict(dictionary, file):
    with open(file, "wb") as f:
        pickle.dump(dictionary, f)


def load_dict(file):
    with open(file, 'rb') as f: 
         return pickle.load(f)


def fix_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    np.random.seed(seed)


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
    y,x = img.shape
    startx = x // 2 - (cropx // 2)
    starty = y // 2 - (cropy // 2)
    return img[starty:starty+cropy, startx:startx+cropx]


def crop_and_scale(img, cropXY):
    y,x = img.shape

    dim = min(x, y)
    img = crop_center(img, (dim, dim))
    
    return cv2.resize(img, cropXY)


def tokens_to_text(tokens):
    return " ".join(tokens).replace(" .", ".")


def create_image_attention_mask(attention_weights, image_size, grid_size, interpolation=cv2.INTER_NEAREST):
    return cv2.resize(attention_weights.reshape(grid_size), image_size, interpolation=interpolation)


def create_avg_image_attention_mask(attention_weights, image_size, grid_size, interpolation=cv2.INTER_NEAREST):
    return np.sum([create_image_attention_mask(weights, image_size, grid_size, interpolation=interpolation) for weights in attention_weights], axis=0) / len(attention_weights)