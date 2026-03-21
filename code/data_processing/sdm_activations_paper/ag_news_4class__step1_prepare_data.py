# Copyright Reexpress AI, Inc. All rights reserved.

import json
import os
import numpy as np

import argparse

import data_utils
from datasets import load_dataset
import random
import uuid

REEXPRESS_ID_KEY = "id"
REEXPRESS_LABEL_KEY = "label"
REEXPRESS_DOCUMENT_KEY = "document"
REEXPRESS_ATTRIBUTES_KEY = "attributes"
REEXPRESS_EMBEDDING_KEY = "embedding"


def preprocess(split_name):
    ds = load_dataset("fancyzhx/ag_news", split=split_name)
    json_list = []
    valid_labels = list(range(4))
    lengths = []
    document_set = set()
    for instance_index in range(len(ds)):
        instance = ds[instance_index]
        new_dict = {}
        if instance["text"] in document_set:
            print("WARNING: DUPLICATE")
            continue
        else:
            document_set.add(instance["text"])
        new_dict[REEXPRESS_ID_KEY] = f"rowid_{instance_index}_{str(uuid.uuid4())}"
        new_dict[REEXPRESS_DOCUMENT_KEY] = instance["text"]
        assert int(instance["label"]) in valid_labels
        new_dict[REEXPRESS_LABEL_KEY] = int(instance["label"])
        lengths.append(len(instance["text"].split()))
        json_list.append(new_dict)
    print(f"Lengths: Mean: {np.mean(lengths)}; min: {np.min(lengths)}; max: {np.max(lengths)}")
    return json_list


def main():
    parser = argparse.ArgumentParser(description="-----[Preprocess data]-----")
    parser.add_argument("--seed_value", default=0, type=int, help="seed_value")
    parser.add_argument("--output_dir", default="",
                        help="")
    options = parser.parse_args()

    random.seed(0)

    split_name = "train"
    json_list = preprocess(split_name)
    print(f"Total training documents: {len(json_list)}")
    random.shuffle(json_list)
    dataset_size = len(json_list) // 2
    train_json_list = json_list[0:dataset_size]
    output_file = os.path.join(options.output_dir, f"train.ag_news.jsonl")
    data_utils.save_json_lines(output_file, train_json_list)

    calibration_json_list = json_list[dataset_size:]
    output_file = os.path.join(options.output_dir, f"calibration.ag_news.jsonl")
    data_utils.save_json_lines(output_file, calibration_json_list)

    split_name = "test"
    json_list = preprocess(split_name)
    output_file = os.path.join(options.output_dir, f"test.ag_news.jsonl")
    data_utils.save_json_lines(output_file, json_list)


if __name__ == "__main__":
    main()
