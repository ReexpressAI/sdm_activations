# Copyright Reexpress AI, Inc. All rights reserved.

import json
import numpy as np

import argparse
import time
import codecs
import copy
import random
import uuid

from datasets import load_dataset

import sdm_network_constants


def save_json_lines(filename_with_path, json_list):
    with codecs.open(filename_with_path, "w", encoding="utf-8") as f:
        for json_dict in json_list:
            f.write(json.dumps(json_dict, ensure_ascii=True) + "\n")


def get_lines(longest_subset_size=2000, min_length=5, max_length=60):

    lengths = []
    lengths_longest_subset = []
    lines = []
    lines_longest_subset = []
    dataset = load_dataset("sentence-transformers/wikipedia-en-sentences")
    dataset = dataset.shuffle(seed=0)
    assert len(dataset["train"]) == len(dataset.unique(column="sentence")["train"]), \
        f'{len(dataset["train"])}, {len(dataset.unique(column="sentence")["train"])}'

    # Sentences between min_length and max_length words
    dataset = dataset["train"].filter(
        lambda x: min_length <= len(x['sentence'].strip().split()) <= max_length
    )
    print(f"Size of dataset after filtering by length: {len(dataset)}")

    # Add a column with sentence lengths
    dataset = dataset.map(
        lambda x: {'length': len(x['sentence'].strip().split())}
    )
    # Sort by the length column
    dataset = dataset.sort('length', reverse=True)
    for x in dataset:
        if len(lines_longest_subset) < longest_subset_size:
            lines_longest_subset.append(x["sentence"].strip())
            lengths_longest_subset.append(len(x["sentence"].strip().split()))
        else:
            lines.append(x["sentence"].strip())
            lengths.append(len(x["sentence"].strip().split()))

    print(f"Lines: {len(lines)}; min: {min(lengths)}; max: {max(lengths)}; mean: {np.mean(lengths)}")
    print(f"Longest subset: {len(lines_longest_subset)}; "
          f"min: {min(lengths_longest_subset)}; max: {max(lengths_longest_subset)}; "
          f"mean: {np.mean(lengths_longest_subset)}")
    return lines, lines_longest_subset


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
    max_attempts = 50
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


def construct_eval_shuffle(options, json_lines):
    """
    In this case, just the multiset and the true document. As such, the label is always 1.

    fully shuffled multiset -> correct ordering -> label 1
    """
    json_objects = []

    for line in json_lines:
        processed_sentence = line.strip()
        # here, we consider up to size 5, ensuring all the same sentences are considered across datasets
        if processed_sentence is None or len(processed_sentence.split()) < 2:
            print(f"SKIPPING: {processed_sentence.split()}")
            continue

        tokens = processed_sentence.split()
        # Check if the last multiset_size words have enough variety for distinct shuffles
        multiset_words = tokens[-options.multiset_size:]
        if len(set(multiset_words)) < 2:  # Not enough unique words to create distinct shuffles
            print(f"SKIPPING (insufficient variety): {processed_sentence}")
            continue

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


def main():
    parser = argparse.ArgumentParser(description="-----[Preprocess data]-----")
    parser.add_argument("--multiset_size", default=3, type=int, help="")
    parser.add_argument("--output_genai_train_file", default="", help="Only multiset and true as document")
    parser.add_argument("--output_genai_calibration_file", default="", help="Only multiset and true as document")
    parser.add_argument("--output_held_out_eval_file", default="", help="Only multiset and true as document")
    parser.add_argument("--output_held_out_challenge_eval_file", default="",
                        help="Only multiset and true as document. Longest documents in subset.")
    parser.add_argument("--output_remaining_lines_file", default="",
                        help="Only multiset and true as document. Remaining documents not assigned a subset.")

    options = parser.parse_args()

    random.seed(0)
    start_time = time.time()

    test_set_size = 2000
    calibration_set_size = 5000
    training_set_size = calibration_set_size
    longest_subset_size = test_set_size
    remaining_to_save = 100000

    all_lines, longest_lines = get_lines(longest_subset_size=longest_subset_size, min_length=5, max_length=60)

    json_lines = construct_eval_shuffle(options, longest_lines)
    save_json_lines(options.output_held_out_challenge_eval_file, json_lines)
    print(f"Held-out challenge eval set lines (ordered by length): {len(json_lines)}")

    random.shuffle(all_lines)

    test_set_json_lines = all_lines[-test_set_size:]
    test_set_json_lines = construct_eval_shuffle(options, test_set_json_lines)
    save_json_lines(options.output_held_out_eval_file, test_set_json_lines)
    print(f"Held-out eval set lines: {len(test_set_json_lines)}")

    all_lines = all_lines[0:-test_set_size]  # reduce
    # calibration set:
    calibration_set_json_lines = all_lines[-calibration_set_size:]
    calibration_set_json_lines = construct_eval_shuffle(options, calibration_set_json_lines)
    save_json_lines(options.output_genai_calibration_file, calibration_set_json_lines)
    print(f"Calibration set lines: {len(calibration_set_json_lines)}")
    # training:
    all_lines = all_lines[0:-calibration_set_size]  # reduce
    training_set_json_lines = all_lines[-training_set_size:]
    training_set_json_lines = construct_eval_shuffle(options, training_set_json_lines)
    save_json_lines(options.output_genai_train_file,
                    training_set_json_lines)
    print(f"Training set lines: {len(training_set_json_lines)}")
    # remaining
    all_lines = all_lines[0:-training_set_size]  # reduce
    remaining_lines = all_lines[-remaining_to_save:]
    if len(remaining_lines) > 0:
        remaining_lines = construct_eval_shuffle(options, remaining_lines)
        save_json_lines(options.output_remaining_lines_file,
                        remaining_lines)
        print(f"Remaining (unassigned) lines up to {remaining_to_save}: {len(remaining_lines)}")

    cumulative_time = time.time() - start_time
    print(f"Cumulative running time: {cumulative_time}")


if __name__ == "__main__":
    main()
