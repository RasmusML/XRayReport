"""Train a model on the X-Ray dataset."""

from utils import *
from dataset import *
from plots import *
from models import *

import stanza
import logging
import argparse
import os
import yaml


def main(args):
    logging.basicConfig(format="%(asctime)s: %(message)s", level=logging.INFO, datefmt="%I:%M:%S")

    fix_seed()

    stanza.download("en")

    REPORT_PATH = "./data/raw/reports"
    IMAGE_PATH = "./data/raw/images"

    metadata = load_reports(REPORT_PATH)

    if args.size > 0:
        metadata = metadata[:args.size]
    
    logging.info("loading reports...")
    reports = prepare_reports(metadata)

    logging.info("loading images...")
    images = load_images(metadata, IMAGE_PATH, resized=(224, 224))[reports.index]

    logging.info("preprocessing...")
    tokenizer = stanza_tokenizer()
    tokenized_reports = reports.apply(lambda text : tokenize(text, tokenizer))

    vocabulary = build_vocabulary([token for tokens in tokenized_reports for token in tokens])
    token2id, _ = map_token_and_id(vocabulary)
    
    model_name = args.model
    
    if args.model == "vit":
        model = XRayViTModel(len(vocabulary))
    elif args.model == "base":
        model = XRayBaseModel(len(vocabulary))
    elif args.model == "playground":
        model = XRayPlaygroundModel(len(vocabulary))
    elif args.model == "chex":
        glove_vector = download_glove()
        word_embeddings = get_word_embeddings(token2id, glove_vector)
        model = CheXNetBaseNet(word_embeddings)
    else:
        raise ValueError(f"model {args.model} not supported")

    processed_images = model.encoder.preprocess(images)

    train_test_split = .9
    train_validation_split = .9

    total_train_size = int(len(images) * train_test_split)
    train_size = int(total_train_size * train_validation_split)

    train_dataset = XRayDataset(processed_images[:train_size], tokenized_reports[:train_size], token2id)
    validation_dataset = XRayDataset(processed_images[train_size:total_train_size], tokenized_reports[train_size:total_train_size], token2id)
    test_dataset = XRayDataset(processed_images[total_train_size:], tokenized_reports[total_train_size:], token2id)

    logging.info("training...")
    train(model_name, model, vocabulary, train_dataset, validation_dataset, args.epochs, args.lr, args.batch_size, args.weight_decay)
    
    logging.info("evaluating...")
    result_path = os.path.join("results", model_name, "result.pkl")
    result = load_dict(result_path)
    result["test_loss"] = evaluate(model, test_dataset, token2id)
    save_dict(result, result_path)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description=__doc__)
    
    parser.add_argument("--model", choices=["vit", "base"])
    parser.add_argument("--size", default=-1, type=int)
    parser.add_argument("--epochs", default=50)
    parser.add_argument("--batch_size", default=64)
    parser.add_argument("--lr", default=0.1)
    parser.add_argument("--weight_decay", default=0.1)
    
    parser.add_argument("--config")

    args = parser.parse_args()

    # override arguments with config file
    if args.config:
        opt = yaml.load(open(args.config), Loader=yaml.FullLoader)

        for k, v in opt.items():
            setattr(args, k, v)

    main(args)

