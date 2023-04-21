"""
Train a model on the X-Ray dataset.
"""

from utils import *
from dataset import *
from plots import *
from models import *
from nlp import *

import stanza
import logging
import argparse
import os
from enum import Enum

import torch
from torch import optim


"""
Supported Configs
-----------------
{
    "data": {
        "size": 1000,
        "preprocessed_images": "data/processed/chex1_images.pt",
        "split": [0.8, 0.1, 0.1] # train, validation, test
    },
    "training": {
        "epochs": 100,
        "batch_size": 32,
        "optimizer": optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5),
        "weighted_loss": True
        "checkpoint_save_freq": 100,
        "bleu_eval_freq": 50,
        "bleu_max_samples": 10
    },
    "model": {
        "embeddings": Embedding.UI_XRAY | ...
    }
}

"""

class Embeddings(Enum):
    IU_XRAY = 1
    GLOVE   = 2


REPORT_PATH = "./data/raw/reports"
IMAGE_PATH = "./data/raw/images"


def main(args):
    logging.basicConfig(format="%(asctime)s: %(message)s", level=logging.INFO, datefmt="%I:%M:%S")

    fix_seed()
    stanza.download("en")

    model_name = args.model

    if model_name == "chex1":
        config = {
            "data": {
                "preprocessed_images": "data/processed/chex1_images.pt",
                "split": [0.8, 0.1, 0.1] # train, validation, test
            },
            "training": {
                "epochs": 800,
                "batch_size": 128,
                "optimizer": lambda params: optim.Adam(params, lr=1e-4, weight_decay=1e-5),
                "weighted_loss": True,
                "checkpoint_save_freq": 5,
                "bleu_eval_freq": 5,
                "bleu_max_samples": 10,
            },
            "model": {
                "embeddings": Embeddings.IU_XRAY
            }

        }

    elif model_name == "chex2":
        config = {
            "data": {
                "preprocessed_images": "data/processed/chex2_images.pt",
                "split": [0.8, 0.1, 0.1] # train, validation, test
            },
            "training": {
                "epochs": 3800,
                "batch_size": 128,
                "optimizer": lambda params: optim.Adam(params, lr=1e-4, weight_decay=1e-4),
                "weighted_loss": True,
                "checkpoint_save_freq": 100,
                "bleu_eval_freq": 50,
                "bleu_max_samples": 200,
            },
            "model": {
                "embeddings": Embeddings.IU_XRAY
            }
        }

    elif model_name == "playground":
        config = {
            "data": {
                "size": 100,
                "split": [0.8, 0.1, 0.1] # train, validation, test
            },
            "training": {
                "epochs": 100,
                "batch_size": 32,
                "optimizer": lambda params: optim.Adam(params, lr=1e-4, weight_decay=0),
                "weighted_loss": True,
                "checkpoint_save_freq": 50,
                "bleu_eval_freq": 50,
                "bleu_max_samples": 10,
            },

            "model": {
            }
        }

    elif model_name == "vit":
        config = {
            "data": {
                "size": 100,
                "split": [0.8, 0.1, 0.1] # train, validation, test
            },
            "training": {
                "epochs": 80,
                "batch_size": 32,
                "optimizer": lambda params: optim.Adam(params, lr=1e-4, weight_decay=1e-5),
                "weighted_loss": True,
                "checkpoint_save_freq": 100,
                "bleu_eval_freq": 50,
                "bleu_max_samples": 200,
            },

            "model": {
            }
        }

    else:
        raise ValueError(f"model {model_name} not supported")
    
    
    logging.info(f"configs: {config}.")

    data_config = config["data"]
    train_config = config["training"]
    model_config = config["model"]

    logging.info("loading reports...")
    metadata = load_metadata(REPORT_PATH)
    shuffle_metadata(metadata, seed=42)
    
    reports = prepare_reports(metadata)

    logging.info("tokenizing...")
    tokenizer = stanza_tokenizer()
    tokenized_reports = reports.apply(lambda text : tokenize(text, tokenizer))

    vocabulary = build_vocabulary([token for tokens in tokenized_reports for token in tokens])
    token2id, id2token = map_token_and_id(vocabulary)

    logging.info(f"loading model {model_name}...")

    if "embeddings" in model_config:
        embeddings = model_config["embeddings"]
        if embeddings == Embeddings.IU_XRAY:
            vectors, embed_dim = load_pubmed_embeddings_IU_xray_pretrained()
        elif embeddings == Embeddings.GLOVE:
            vectors, embed_dim = download_glove()
        else:
            raise ValueError(f"embeddings {embeddings} not supported")
        
        word_embeddings = prepare_word_embeddings(token2id, vectors, embed_dim)

    # load models
    if model_name == "chex1":
        model = CheXNet1(word_embeddings)

    elif model_name == "chex2":
        model = CheXTransformerNet(word_embeddings)

    elif model_name == "playground":
        model = PlaygroundModel(len(vocabulary))

    elif model_name == "vit":
        model = XRayViTModel(word_embeddings)

    else:
        raise ValueError(f"model {model_name} not supported")
    

    optimizer = train_config["optimizer"](model.parameters())


    if "size" in data_config:
        data_size = min(data_config["size"], len(reports))
    else:
        data_size = len(reports)

    if "bleu_max_samples" in train_config:
        bleu_max_samples = train_config["bleu_max_samples"]
    else:
        bleu_max_samples = -1

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
    test_dataset = XRayDataset(images[test_start:test_end], tokenized_reports[test_start:test_end], token2id)

    logging.info(f"train dataset size: {len(train_dataset)}")
    logging.info(f"validation dataset size: {len(validation_dataset)}")
    logging.info(f"test dataset size: {len(test_dataset)}")

    if train_config["weighted_loss"]:
        token_occurencies = count_token_occurences(tokenized_reports)
        loss_weights = torch.zeros(len(vocabulary), dtype=torch.float32)
        loss_weights[token2id["<END>"]] = 1 / len(tokenized_reports)

        for token, occurencies in token_occurencies.items():
            loss_weights[token2id[token]] = 1 / occurencies

    else:
        loss_weights = None

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    logging.info("training...")
    train(model_name, model, vocabulary, train_dataset, validation_dataset,
          batch_size=train_config["batch_size"], epochs=train_config["epochs"], optimizer=optimizer, 
          loss_weights=loss_weights, checkpoint_save_freq=train_config["checkpoint_save_freq"], bleu_eval_freq=train_config["bleu_eval_freq"], 
          bleu_max_samples=bleu_max_samples, 
          device=device)
    

    logging.info("computing BLEU score for test set")
    test_bleu = compute_bleu(model, test_dataset, token2id, id2token, device=device)
    logging.info(f"test BLEU score: {test_bleu}")
    save_dict({ "bleu": test_bleu }, os.path.join("results", model_name, "test_result.pkl"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model")

    args = parser.parse_args()

    main(args)

