"""
Create pairs of true report and generated report, and save them to later feed into the COCO evaluation pipeline.
"""

import argparse
import logging
import os
import torch
import json

from utils import *
from dataset import *
from models import *
from nlp import *

from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.cider.cider import Cider
from pycocoevalcap.rouge.rouge import Rouge
from pycocoevalcap.meteor.meteor import Meteor
from pycocoevalcap.spice.spice import Spice

REPORT_PATH = "./data/raw/reports"
IMAGE_PATH = "./data/raw/images"


"""
Supported Configs
-----------------
{
    "size": 1000,
    "preprocessed_images": "data/processed/chex1_images.pt",
    "split": [0.8, 0.1, 0.1] # train, validation, test
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
    token2id, id2token = map_token_and_id(vocabulary)

    logging.info(f"loading model {model_name}...")

    # load models
    if model_name == "chex1":
        glove_vector = download_glove()
        word_embeddings = get_word_embeddings(token2id, glove_vector)
        model = CheXNet1(word_embeddings)

        config = {
            "size": 10,
            "preprocessed_images": "data/processed/chex1_images.pt",
            "split": [0.8, 0.1, 0.1] # train, validation, test
        }

    elif model_name == "chex2":
        glove_vector = download_glove()
        word_embeddings = get_word_embeddings(token2id, glove_vector)
        model = CheXTransformerNet(word_embeddings)

        config = {
            "preprocessed_images": "data/processed/chex2_images.pt",
            "split": [0.8, 0.1, 0.1] # train, validation, test
        }

    elif model_name == "playground":
        model = PlaygroundModel(len(vocabulary))
        config = { 
            "size": 100, 
            "split": [0.8, 0.1, 0.1] 
        }

    elif model_name == "vit":
        model = XRayViTModel(len(vocabulary))
        config = { 
            "size": 100, 
            "split": [0.8, 0.1, 0.1] 
        }

    else:
        raise ValueError(f"model {model_name} not supported")
    

    model.load_state_dict(torch.load(os.path.join("models", model_name, "model.pt")))

    logging.info(f"configs: {config}.")

    if "size" in config:
        data_size = min(config["size"], len(reports))
    else:
        data_size = len(reports)

    # some reports may have been filtered due to empty field in the report.
    # Thus, use the index to filter the images
    image_indices = reports[:data_size]
    if  "preprocessed_images" in config:
        images = torch.load(config["preprocessed_images"])
        images = images[image_indices.index]
    else:
        images = load_images(metadata, IMAGE_PATH, resized=(224, 224))
        images = images[image_indices.index]
        images = model.preprocess(images)

    train_split, validation_split, test_split = config["split"]
    assert train_split + validation_split + test_split == 1

    test_start, test_end = int(data_size * (train_split + validation_split)), data_size
    test_dataset = XRayDataset(images[test_start:test_end], tokenized_reports[test_start:test_end], token2id)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    logging.info("creating pairs...")
    references, candidates = prepare_for_evaluation(model, test_dataset, token2id, id2token, device=device, early_exit=-1)

    """ @TODO: maybe re-introduce this when visualizing interesting examples
    data = []
    for i, (reference, candidate) in enumerate(zip(references, candidates)):
        data.append({
            #"image_id": i,
            "test": candidate,
            "refs": reference,
        })

    logging.info("writing to file...")
    ensure_dir(os.path.join("dump", model_name))

    with open(os.path.join("dump", model_name, f"{model_name}_coco_eval.json"), "w") as f:
        json.dump(data, f)
    """

    candidates = [" ".join(candidate) for candidate in candidates]
    references = [" ".join(ref[0]) for ref in references]

    refs = {i: [ref] for i, ref in enumerate(references)}
    hyps = {i: [candidate] for i, candidate in enumerate(candidates)}

    scores = [
        (Bleu(4), ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"]),
        (Meteor(),"METEOR"),
        (Rouge(), "ROUGE_L"),
        (Cider(), "CIDEr"),
        #(Spice(), "SPICE")
    ]

    for score, method in scores:
        logging.info(f"computing {method} score...")

        score, scores = score.compute_score(hyps, refs)
        if type(method) == list:
            for m, s in zip(method, score):
                print(f"{m}: {s}")
        else:
            print(f"{method}: {score}")
    
    logging.info("done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model")

    args = parser.parse_args()
    
    main(args)