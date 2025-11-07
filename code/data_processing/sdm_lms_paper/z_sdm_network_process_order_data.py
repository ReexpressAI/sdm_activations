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
import random
import uuid

import sdm_network_constants


def save_json_lines(filename_with_path, json_list):
    with codecs.open(filename_with_path, "w", encoding="utf-8") as f:
        for json_dict in json_list:
            f.write(json.dumps(json_dict, ensure_ascii=True) + "\n")


def get_lines(filepath_with_name):
    lines = []
    with codecs.open(filepath_with_name, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            lines.append(line)
    return lines


def get_sentence(line):
    if line == "":
        return None

    # Find the position of the first period
    period_pos = line.find('.')

    # If no period found or period is at the start, return False
    if period_pos <= 0:
        return None

    # Extract the part before the period
    before_period = line[:period_pos]

    # Check if it's a valid integer
    try:
        int(before_period)
        return line[period_pos+1:].strip()
    except ValueError:
        return None


def get_formatted_genai_response_from_tokens(sentence, is_correct=True, noise_tag_probability=0.1):
    if is_correct:
        genai_response = f"<sentence>{sentence}</sentence>\n<verified>Yes</verified>"
        verification_input = f"<sentence>{sentence}</sentence>\n<verified>"
    else:
        if random.random() < noise_tag_probability:
            tags_to_noise_type = [0, 1, 2]
            random.shuffle(tags_to_noise_type)
            tags_to_noise = tags_to_noise_type[0]
            if tags_to_noise == 0:  # remove left sentence tag
                verification_input = f"{sentence}</sentence>\n<verified>"
                genai_response = verification_input + f"No</verified>"
            elif tags_to_noise == 1:  # remove right sentence tag
                verification_input = f"<sentence>{sentence}\n<verified>"
                genai_response = verification_input + f"No</verified>"
            elif tags_to_noise == 2:  # remove both left and right sentence tags
                verification_input = f"{sentence}\n<verified>"
                genai_response = verification_input + f"No</verified>"
            else:
                assert False
        else:
            genai_response = f"<sentence>{sentence}</sentence>\n<verified>No</verified>"
            verification_input = f"<sentence>{sentence}</sentence>\n<verified>"
    return genai_response, verification_input


def get_two_distinct_shuffles(original_order) -> list[str]:
    assert len(original_order) > 2
    partial_shuffle_string1 = ""
    partial_shuffle_string2 = ""
    original_sentence = " ".join(original_order)
    partial_shuffle_set = set([])
    max_attempts = 1000
    while len(original_order) > 1 and \
            (partial_shuffle_string1 == original_sentence or
             partial_shuffle_string2 == original_sentence or
             len(partial_shuffle_set) != 2):  # ensure mismatches
        partial_shuffle_string1 = " ".join(get_complete_shuffle(original_order=" ".join(original_order).split()))
        partial_shuffle_string2 = " ".join(get_complete_shuffle(original_order=" ".join(original_order).split()))
        partial_shuffle_set = set([partial_shuffle_string1, partial_shuffle_string2])
        max_attempts -= 1
        if max_attempts <= 0:
            print(f"WARNING: Max shuffling iterations reached. Number of unique partial shuffles found: "
                  f"{len(partial_shuffle_set)}")
            break
    return list(partial_shuffle_set)


def get_complete_shuffle(original_order):
    assert len(original_order) > 1
    tokens = copy.deepcopy(original_order)
    original_order = copy.deepcopy(original_order)
    while len(tokens) > 1 and tokens == original_order:  # ensure mismatch
        random.shuffle(tokens)
    return tokens


def construct_prompt(sentence_string, multiset_size):
    sentence_tokens = sentence_string.split()
    first = sentence_tokens[0: -multiset_size]
    second = sentence_tokens[-multiset_size:]
    sentence_prefix = " ".join(first)
    two_shuffled_suffix_strings = get_two_distinct_shuffles(second)
    assert len(two_shuffled_suffix_strings) == 2 and two_shuffled_suffix_strings[0] != "" and two_shuffled_suffix_strings[1] != ""
    multiset = two_shuffled_suffix_strings[0]
    multiset_for_synthetic_negative = two_shuffled_suffix_strings[1]

    optional_hint_text = ""  # always blank, but included here for reference on how it is constructed at test-time
    prompt_text = f"Complete the sentence '{sentence_prefix}' by reordering all of the following without adding new punctuation nor words: '{multiset}'. Only reply with the sentence in the XML <sentence> </sentence> followed by <verified>Yes</verified> if your answer correctly addressed the instructions, and <verified>No</verified> if it did not.{optional_hint_text}"

    # construct synthetic negative using a different shuffle than that of the prompt:
    first.extend(multiset_for_synthetic_negative.split())
    synthetic_negative_sentence = " ".join(first)

    return prompt_text, multiset, synthetic_negative_sentence


def construct_eval_shuffle(json_lines):
    """
    In this case, just the multiset and the true document. As such, the label is always 1.

    fully shuffled multiset -> correct ordering -> label 1
    """
    json_objects = []

    for line in json_lines:
        line = line.strip()
        processed_sentence = get_sentence(line)
        if processed_sentence is None or len(processed_sentence.split()) < 2:
            continue
        tokens = processed_sentence.split()

        original_order = copy.deepcopy(tokens)
        original_sentence = " ".join(original_order)
        genai_response, _ = \
            get_formatted_genai_response_from_tokens(original_sentence, is_correct=True)

        prompt, multiset, synthetic_negative_sentence = \
            construct_prompt(original_sentence, multiset_size=options.multiset_size)

        assert synthetic_negative_sentence != original_sentence
        synthetic_negative_response, _ = \
            get_formatted_genai_response_from_tokens(synthetic_negative_sentence, is_correct=False)

        json_obj = {}
        json_obj[sdm_network_constants.REEXPRESS_ID_KEY] = str(uuid.uuid4())
        # The original reference sentence without shuffling:
        json_obj[sdm_network_constants.ORIGINAL_DOCUMENT_ORDER_KEY] = original_sentence
        # For input to generative models as the assistant output:
        json_obj[sdm_network_constants.REEXPRESS_GENAI_DOCUMENT_KEY] = genai_response  # in this case, with the correct ordering
        # Cache prompt:
        json_obj[sdm_network_constants.CACHED_PROMPT_KEY] = prompt
        json_obj[sdm_network_constants.DEFAULT_NEGATIVE_KEY] = synthetic_negative_response
        # for reference for analysis to avoid parsing the prompt
        json_obj[sdm_network_constants.MULTISET_KEY] = multiset

        json_objects.append(json_obj)

    return json_objects


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="-----[Preprocess data]-----")
    parser.add_argument("--multiset_size", default=3, type=int, help="")
    parser.add_argument("--input_file", default="", help="")
    parser.add_argument("--output_genai_train_file", default="", help="Only multiset and true as document")
    parser.add_argument("--output_genai_calibration_file", default="", help="Only multiset and true as document")
    parser.add_argument("--output_held_out_eval_file", default="", help="Only multiset and true as document")
    parser.add_argument("--output_all_combined_eval_file", default="", help="Only multiset and true as document")

    options = parser.parse_args()

    random.seed(0)
    start_time = time.time()
    all_lines = get_lines(options.input_file)
    random.shuffle(all_lines)

    test_set_json_lines = all_lines[-250:]
    test_set_json_lines = construct_eval_shuffle(test_set_json_lines)
    save_json_lines(options.output_held_out_eval_file, test_set_json_lines)
    print(f"Held-out eval set lines: {len(test_set_json_lines)}")

    remaining_json_lines = all_lines[0:-250]
    # split:
    partition_size = len(remaining_json_lines) // 3
    # 1/3 as the calibration set:
    calibration_set_json_lines = remaining_json_lines[-partition_size:]
    calibration_set_json_lines = construct_eval_shuffle(calibration_set_json_lines)
    save_json_lines(options.output_genai_calibration_file, calibration_set_json_lines)
    print(f"Calibration set lines: {len(calibration_set_json_lines)}")
    # remaining for training:
    training_set_json_lines = remaining_json_lines[0:-partition_size]
    # In this first pass, we only save the correct generation, as with the eval sets. Then we generate for
    # hard negatives in the training script.
    training_set_json_lines = construct_eval_shuffle(training_set_json_lines)
    save_json_lines(options.output_genai_train_file,
                    training_set_json_lines)
    print(f"Training set lines: {len(training_set_json_lines)}")

    all_combined_json_lines = test_set_json_lines + calibration_set_json_lines + training_set_json_lines
    save_json_lines(options.output_all_combined_eval_file,
                    all_combined_json_lines)
    print(f"Combined lines: {len(all_combined_json_lines)}")

    cumulative_time = time.time() - start_time
    print(f"Cumulative running time: {cumulative_time}")
