import xml.etree.ElementTree as ET
import pandas as pd
from collections import defaultdict
import os
import numpy as np
from PIL import Image

from torch.utils.data import DataLoader, Dataset
import random

import spacy
import stanza

import logging

from utils import *


def load_metadata(path):
    reportFeatures = defaultdict(list)

    filenames = get_filenames(path)
    logging.info(f"found {len(filenames)} reports.")

    for i, filename in enumerate(filenames):
        tree = ET.parse(os.path.join(path, filename))
        root = tree.getroot()

        medlineCitationTag = root.find("MedlineCitation")
        assert medlineCitationTag != None, "Failed to find MedlineCitation tag"

        articleTag = medlineCitationTag.find("Article")
        assert articleTag != None, "Failed to find Article tag"

        abstractTag = articleTag.find("Abstract")
        assert abstractTag != None, "Failed to find Abstract tag"

        parentImageTags = root.findall("parentImage")

        for parentImageTag in parentImageTags:
            reportFeatures["patient_id"].append(i)
            reportFeatures["image_name"].append(parentImageTag.attrib["id"] + ".png")

            for reportCategory in abstractTag:
                feature = reportCategory.attrib["Label"].lower()
                reportFeatures[feature].append(reportCategory.text)

    return pd.DataFrame(reportFeatures)


def shuffle_metadata(metadata, seed=42):
    random.seed(seed)

    groups = [metadata for _, metadata in metadata.groupby('patient_id')]
    random.shuffle(groups)

    metadata = pd.concat(groups).reset_index(drop=True)

    return metadata


def load_images(metadata, image_path, resized=(224, 224)):
    raw_images = {}

    for sample in metadata["image_name"]:
        image = load_png(os.path.join(image_path, sample))
        raw_images[sample] = crop_and_scale(image[..., 0], resized)

    images = torch.tensor(np.array(list(raw_images.values())), dtype=torch.float32)
    normalized_images = images / 255.

    return normalized_images


def prepare_reports(metadata):
    reports = metadata["findings"].astype(str) + "\n" + metadata["impression"].astype(str)
    logging.info(f"raw report length: {len(reports)}")

    reports.replace("[N|n]one", "", regex=True, inplace=True)
    reports = reports[reports != "\n"] # if both are None, then only "\n" remains
    
    reports = reports.rename("report")
    logging.info(f"post-processing report length: {len(reports)}")

    return reports


#
# prepare for the models
#

def stanza_tokenizer():
    nlp = stanza.Pipeline(lang="en", processors="tokenize")
    return lambda text: [token.text for sentence in nlp(text).sentences for token in sentence.tokens]


def spacy_tokenizer():
    nlp = spacy.load('en_core_web_lg')
    return lambda text: [token.text for token in nlp(text)]


def tokenize(text, tokenizer):

    def valid_token(token):
        return (token.isalpha() or token == ".") and (not token == "xxxx")

    text = text.lower()
    tokens = tokenizer(text)

    return [token for token in tokens if valid_token(token)]


def build_vocabulary(tokens):
    return set(["<UNK>", "<PAD>", "<START>", "<END>"]) | set(tokens)


def map_token_and_id(vocabulary):
    stabil_vocabulary = list(vocabulary)
    stabil_vocabulary.sort()
    return {token: i for i, token in enumerate(stabil_vocabulary)}, {i: token for i, token in enumerate(stabil_vocabulary)}


def count_token_occurences(tokenized_reports):
    token_counts = {}

    for token_ids in tokenized_reports:
        for token_id in token_ids:
            if token_id in token_counts:
                token_counts[token_id] += 1
            else:
                token_counts[token_id] = 1
                
    return token_counts