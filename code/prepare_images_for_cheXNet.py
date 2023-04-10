"""Train a model on the X-Ray dataset."""

from utils import *
from dataset import *
from plots import *
from models import *

import logging
import os
import argparse

def main(args):
    logging.basicConfig(format="%(asctime)s: %(message)s", level=logging.INFO, datefmt="%I:%M:%S")

    REPORT_PATH = "./data/raw/reports"
    IMAGE_PATH = "./data/raw/images"

    metadata = load_reports(REPORT_PATH)

    if args.size > 0:
        metadata = metadata[:args.size]

    reports = prepare_reports(metadata)

    raw_images = load_images(metadata, IMAGE_PATH, resized=(224, 224))[reports.index]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    encoder = CheXNetEncoder()
    encoder = encoder.to(device)

    images = encoder.preprocess(raw_images)
    
    os.makedirs("data/processed", exist_ok=True)
    torch.save(images, "data/processed/chex_images.pt")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--size", default=-1, type=int)

    args = parser.parse_args()
    
    main(args)

