import xml.etree.ElementTree as ET
import pandas as pd
from collections import defaultdict
import os
import numpy as np
from PIL import Image

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


def load_images(metadata, image_path, resized=(256, 256)):
    raw_images = {}

    for sample in metadata["image_name"]:
        image = load_png(os.path.join(image_path, sample))
        image = crop_and_scale(image, resized)
        raw_images[sample] = image

    return np.array(list(raw_images.values()))
