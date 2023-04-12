"""
Train a model on the X-Ray dataset.
"""

from utils import *
from dataset import *
from plots import *
from models import *

import stanza
import logging
import argparse
import os

import torch
from torch import optim


REPORT_PATH = "./data/raw/reports"
IMAGE_PATH = "./data/raw/images"

"""
Supported Configs
-----------------
{
    "data": {
        "size": 10,
        "preprocessed_images": "data/processed/chex1_images.pt",
        "split": [0.8, 0.1, 0.1] # train, validation, test
    },
    "training": {
        "epochs": 100,
        "batch_size": 32,
        "optimizer": optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5),
        "weighted_loss": True
        "checkpoint_save_freq": 100
    },
}

"""

def main(args):
    logging.basicConfig(format="%(asctime)s: %(message)s", level=logging.INFO, datefmt="%I:%M:%S")

    fix_seed()
    stanza.download("en")

    model_name = args.model

    logging.info("loading reports...")
    metadata = load_reports(REPORT_PATH)
    reports = prepare_reports(metadata)

    logging.info("tokenizing...")
    tokenizer = stanza_tokenizer()
    tokenized_reports = reports.apply(lambda text : tokenize(text, tokenizer))

    vocabulary = build_vocabulary([token for tokens in tokenized_reports for token in tokens])
    token2id, _ = map_token_and_id(vocabulary)

    logging.info(f"loading model {model_name}...")

    # load models
    if model_name == "chex1":
        glove_vector = download_glove()
        word_embeddings = get_word_embeddings(token2id, glove_vector)
        model = CheXNet1(word_embeddings)

        config = {
            "data": {
                "size": 10,
                "preprocessed_images": "data/processed/chex1_images.pt",
                "split": [0.8, 0.1, 0.1] # train, validation, test
            },
            "training": {
                "epochs": 100,
                "batch_size": 32,
                "optimizer": optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5),
                "weighted_loss": True,
                "checkpoint_save_freq": 200,
            },
        }

    elif model_name == "chex2":
        glove_vector = download_glove()
        word_embeddings = get_word_embeddings(token2id, glove_vector)
        model = CheXTransformerNet(word_embeddings)

        config = {
            "data": {
                "preprocessed_images": "data/processed/chex2_images.pt",
                "split": [0.8, 0.1, 0.1] # train, validation, test
            },
            "training": {
                "epochs": 2400,
                "batch_size": 128,
                "optimizer": optim.Adam(model.parameters(), lr=0.0001),
                "weighted_loss": True,
                "checkpoint_save_freq": 100,
            },
        }
    elif model_name == "playground":
        model = PlaygroundModel(len(vocabulary))

        config = {
            "data": {
                "size": 100,
                "split": [0.8, 0.1, 0.1] # train, validation, test
            },
            "training": {
                "epochs": 1000,
                "batch_size": 32,
                "optimizer": optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5),
                "weighted_loss": True,
                "checkpoint_save_freq": 100,
            },
        }

    elif model_name == "vit":
        model = XRayViTModel(len(vocabulary))

        config = {
            "data": {
                "size": 100,
                "split": [0.8, 0.1, 0.1] # train, validation, test
            },
            "training": {
                "epochs": 1000,
                "batch_size": 32,
                "optimizer": optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5),
                "weighted_loss": True,
                "checkpoint_save_freq": 100,
            },
        }

    else:
        raise ValueError(f"model {model_name} not supported")
    

    logging.info(f"configs: {config}.")

    data_config = config["data"]
    train_config = config["training"]

    if "size" in data_config:
        data_size = min(data_config["size"], len(reports))
    else:
        data_size = len(reports)

    # some reports may have been filtered due to empty field in the report.
    # Thus, use the index to filter the images
    image_indices = reports[:data_size]
    if  "preprocessed_images" in data_config:
        images = torch.load(data_config["preprocessed_images"])
        images = images[image_indices.index]
    else:
        images = load_images(metadata, IMAGE_PATH, resized=(224, 224))
        images = images[image_indices.index]
        images = model.preprocess(images)

    train_split, validation_split, test_split = data_config["split"]
    assert train_split + validation_split + test_split == 1

    train_start, train_end = 0, int(data_size * train_split)
    validation_start, validation_end = train_end, int(data_size * (train_split + validation_split))
    test_start, test_end = validation_end, data_size

    train_dataset = XRayDataset(images[train_start:train_end], tokenized_reports[train_start:train_end], token2id)
    validation_dataset = XRayDataset(images[validation_start:validation_end], tokenized_reports[validation_start:validation_end], token2id)
    test_dataset = XRayDataset(images[test_start:test_end], tokenized_reports[test_start:test_end], token2id) # @TODO: WIP

    if train_config["weighted_loss"]:
        token_occurencies = count_token_occurences(tokenized_reports)
        loss_weights = torch.zeros(len(vocabulary), dtype=torch.float32)
        loss_weights[token2id["[END]"]] = 1 / len(tokenized_reports)

        for token, occurencies in token_occurencies.items():
            loss_weights[token2id[token]] = 1 / occurencies



    logging.info("training...")
    train(model_name, model, vocabulary, train_dataset, validation_dataset, batch_size=train_config["batch_size"], epochs=train_config["epochs"], optimizer=train_config["optimizer"], loss_weights=loss_weights, checkpoint_save_freq=train_config["checkpoint_save_freq"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model")

    args = parser.parse_args()

    main(args)

