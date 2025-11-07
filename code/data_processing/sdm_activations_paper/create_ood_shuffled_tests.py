# Copyright Reexpress AI, Inc. All rights reserved.

from typing import Any, Callable, List, Tuple
import json
import os
import numpy as np

import argparse
import time
from pathlib import Path
import codecs
import copy
import data_utils
import random


REEXPRESS_ID_KEY = "id"
REEXPRESS_LABEL_KEY = "label"
REEXPRESS_DOCUMENT_KEY = "document"
REEXPRESS_ATTRIBUTES_KEY = "attributes"
REEXPRESS_EMBEDDING_KEY = "embedding"

FACTCHECK_DATA = "factcheck"
SENTIMENT_DATA = "sentiment"


def get_existing_ids(filepath_with_name):
    existing_ids = set()
    with codecs.open(filepath_with_name, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            json_obj = json.loads(line)
            existing_ids.add(json_obj[REEXPRESS_ID_KEY])
    return existing_ids


def construct_shuffled_streaming(options):
    output_file = options.output_file
    if Path(output_file).exists():
        existing_ids = get_existing_ids(output_file)
    else:
        existing_ids = set()

    instance_i = -1
    with codecs.open(options.input_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            json_obj = json.loads(line)
            instance_i += 1
            if instance_i % 50000 == 0:
                print(f"Currently processing instance {instance_i}")
            if json_obj[REEXPRESS_ID_KEY] in existing_ids:
                continue
            document_tokens = json_obj[REEXPRESS_DOCUMENT_KEY].strip().split()
            original_order = copy.deepcopy(document_tokens)
            while len(document_tokens) > 1 and document_tokens == original_order:
                random.shuffle(document_tokens)
            json_obj[REEXPRESS_DOCUMENT_KEY] = " ".join(document_tokens)
            if options.dataset == FACTCHECK_DATA:
                # ideally we want these instances all rejected as OOD, but as a first approximation, the factcheck
                # label is set to 0
                json_obj[REEXPRESS_LABEL_KEY] = 0
            data_utils.save_by_appending_json_lines(output_file, [json_obj])
            existing_ids.add(json_obj[REEXPRESS_ID_KEY])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="-----[Add embedding data to JSON objects]-----")
    parser.add_argument("--input_file", default="", help="")
    parser.add_argument("--dataset", default=SENTIMENT_DATA, help="")
    parser.add_argument("--output_file", default="", help="")

    options = parser.parse_args()

    assert options.dataset in [FACTCHECK_DATA, SENTIMENT_DATA]

    random.seed(0)
    start_time = time.time()
    construct_shuffled_streaming(options)
    cumulative_time = time.time() - start_time
    print(f"Cumulative running time: {cumulative_time}")
