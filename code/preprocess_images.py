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

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    raw_images = load_images(metadata, IMAGE_PATH, resized=(224, 224))
    raw_images = raw_images.to(device)

    if args.model == "chex1":
        encoder = CheXNetEncoder1()
    elif args.model == "chex2":
        encoder = CheXNetEncoder2()
    else:
        raise ValueError("Unknown model name")

    encoder = encoder.to(device)
    images = process_to_fixed_context(encoder, raw_images)

    os.makedirs("data/processed", exist_ok=True)
    torch.save(images, f"data/processed/{args.model}_images.pt")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument("--model")
    parser.add_argument("--size", default=-1, type=int)

    args = parser.parse_args()
    
    main(args)
