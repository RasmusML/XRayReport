"""Prepare prediction for evaluation."""

from utils import *
from dataset import *
from plots import *
from models import *

import logging
import os
import argparse

def main():
    logging.basicConfig(format="%(asctime)s: %(message)s", level=logging.INFO, datefmt="%I:%M:%S")

    fix_seed()

    stanza.download("en")

    REPORT_PATH = "./data/raw/reports"
    IMAGE_PATH = "./data/raw/images"

    metadata = load_reports(REPORT_PATH)
    metadata_subset = metadata

    logging.info("loading reports...")
    reports_subset = prepare_reports(metadata_subset)
    reports = prepare_reports(metadata)

    logging.info("preprocessing...")
    tokenizer = stanza_tokenizer()
    tokenized_reports = reports.apply(lambda text : tokenize(text, tokenizer))

    vocabulary = build_vocabulary([token for tokens in tokenized_reports for token in tokens])
    token2id, _ = map_token_and_id(vocabulary)
    
    model_name = "chex2"
    images = torch.load("data/processed/chex2_images.pt")[reports_subset.index]

    glove_vector = download_glove()
    word_embeddings = get_word_embeddings(token2id, glove_vector)
    model = CheXTransformerNet(word_embeddings)

    train_test_split = .9
    train_validation_split = .9

    total_train_size = int(len(images) * train_test_split)

    test_dataset = XRayDataset(images[total_train_size:], tokenized_reports[total_train_size:], token2id)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    os.makedirs(os.path.join("results", model_name), exist_ok=True)

    token2id, _ = map_token_and_id(vocabulary)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    result = {}

    # prepare dataloaders
    collate_fn = lambda input: report_collate_fn(token2id["[PAD]"], input)

    train_dl = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)


    # @TODO: beam search



if __name__ == "__main__":
    main()
