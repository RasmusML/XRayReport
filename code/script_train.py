"""Train a model on the X-Ray dataset."""

from utils import *
from dataset import *
from plots import *
from models import *

import stanza
import logging
import argparse

def main(args):
    logging.basicConfig(format="%(asctime)s: %(message)s", level=logging.INFO, datefmt="%I:%M:%S")

    fix_seed()

    stanza.download("en")

    REPORT_PATH = "./data/raw/reports"
    IMAGE_PATH = "./data/raw/images"

    metadata = load_reports(REPORT_PATH)

    if args.size > 0:
        metadata = metadata[:args.size]
    
    reports = prepare_reports(metadata)

    np_images = load_images(metadata, IMAGE_PATH, resized=(256, 256))
    images = torch.tensor(np_images[reports.index])
    images = normalize_images(images)

    tokenizer = stanza_tokenizer()
    tokenized_reports = reports.apply(lambda text : tokenize(text, tokenizer))

    vocabulary = build_vocabulary([token for tokens in tokenized_reports for token in tokens])
    token2id, _ = map_token_and_id_fn(vocabulary)

    train_split = .9
    train_size = int(len(images) * train_split)

    train_dataset = XRayDataset(images[:train_size], tokenized_reports[:train_size], token2id)
    validation_dataset = XRayDataset(images[train_size:], tokenized_reports[train_size:], token2id)

    model_name = "base"
    model = XRayBaseModel(len(vocabulary))

    train(model_name, model, vocabulary, train_dataset, validation_dataset)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--size", default=-1, type=int)

    args = parser.parse_args()

    main(args)