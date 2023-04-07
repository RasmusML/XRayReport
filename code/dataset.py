import xml.etree.ElementTree as ET
import pandas as pd
from collections import defaultdict
import os
import numpy as np
from PIL import Image

from torch.utils.data import DataLoader, Dataset

import spacy
import stanza

import logging

from utils import *


def load_reports(path):
    reportFeatures = defaultdict(list)

    filenames = get_filenames(path)
    logging.info(f"found {len(filenames)} reports.")

    for filename in filenames:
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
            reportFeatures["image_name"].append(parentImageTag.attrib["id"] + ".png")

            for reportCategory in abstractTag:
                feature = reportCategory.attrib["Label"].lower()
                reportFeatures[feature].append(reportCategory.text)

    return pd.DataFrame(reportFeatures)


def load_images(metadata, image_path, resized=(224, 224)):
    raw_images = {}

    for sample in metadata["image_name"]:
        image = load_png(os.path.join(image_path, sample))
        raw_images[sample] = crop_and_scale(image[..., 0], resized)

    images = torch.tensor(np.array(list(raw_images.values())), dtype=torch.float32)
    normalized_images = normalize_images(images)

    return normalized_images


def prepare_reports(metadata):
    reports = metadata["findings"].astype(str) + "\n" + metadata["impression"].astype(str)
    logging.info(f"raw report length: {len(reports)}")

    reports.replace("[N|n]one", "", regex=True, inplace=True)
    #reports.replace("XXXX", "", regex=True, inplace=True)
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
        return token.isalpha() or token == "."

    text = text.lower()
    tokens = tokenizer(text)

    return [token for token in tokens if valid_token(token)]


def build_vocabulary(tokens):
    return set(["[UNK]", "[PAD]", "[START]", "[END]"]) | set(tokens)


def map_token_and_id(vocabulary):
    stabil_vocabulary = list(vocabulary)
    return {token: i for i, token in enumerate(stabil_vocabulary)}, {i: token for i, token in enumerate(stabil_vocabulary)}


def map_token_and_id_fn(vocabulary):
    token2id, id2token = map_token_and_id(vocabulary)
    return lambda token: token2id[token] if token in token2id else token2id["[UNK]"], lambda id: id2token[id]


class XRayDataset(Dataset):
    def __init__(self, images, reports, token2id):
        self.images = images
        self.reports = reports
        self.token2id = token2id

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        report = self.reports.iloc[idx]
        report = ["[START]"] + report + ["[END]"]

        report_length = len(report)
        report_ids = [self.token2id(token) for token in report]

        return image, report_ids, report_length


def report_collate_fn(pad_id, input):
    images, reports, report_lengths = zip(*input)

    report_max_length = max(report_lengths)
    padded_reports = [report + [pad_id] * (report_max_length - length) for report, length in zip(reports, report_lengths)]

    t_images = torch.stack(list(images), dim=0)
    t_reports = torch.tensor(padded_reports)
    t_report_lengths = torch.tensor(report_lengths)

    return t_images, t_reports, t_report_lengths


def normalize_images(images):
    return images.type(torch.float32) / 255.

