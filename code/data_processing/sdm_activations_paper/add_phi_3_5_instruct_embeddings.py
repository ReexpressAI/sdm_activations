# Copyright Reexpress AI, Inc. All rights reserved.

from typing import Any, Callable, List, Tuple
import json
import os
import numpy as np

import argparse
import time
from pathlib import Path
import codecs

import torch

import data_utils
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed


REEXPRESS_ID_KEY = "id"
REEXPRESS_LABEL_KEY = "label"
REEXPRESS_DOCUMENT_KEY = "document"
REEXPRESS_ATTRIBUTES_KEY = "attributes"
REEXPRESS_EMBEDDING_KEY = "embedding"

EXPECTED_EMBEDDING_SIZE = 6144


FACTCHECK_DATA = "factcheck"
SENTIMENT_DATA = "sentiment"


def print_summary(header_label, list_to_process, total=None):
    if total is not None and total > 0:
        print(
            f"{header_label} \tmean: {np.mean(list_to_process) if len(list_to_process) > 0 else 0}, "
            f"\tout of {len(list_to_process)} "
            f"\t({len(list_to_process)/total})% of {total}")
    else:
        print(
            f"{header_label} \tmean: {np.mean(list_to_process) if len(list_to_process) > 0 else 0}, "
            f"\tout of {len(list_to_process)}")


def get_agreement_model_embedding(model, tokenizer, document_text: str, device):
    conv = [{"role": "system", "content": "You are a helpful AI assistant."},
            {"role": "user",
             "content": document_text}]
    input_ids = tokenizer.apply_chat_template(conv, return_tensors="pt", thinking=False,
                                              return_dict=True, add_generation_prompt=True).to(device)
    outputs = model.generate(
        **input_ids,
        max_new_tokens=1,
        output_hidden_states=True,
        return_dict_in_generate=True,
        output_scores=True,
    )
    hidden_states = outputs.hidden_states
    scores = outputs.scores
    # Here, the starting no has a prefix underscore. Other models may differ. tokenizer.convert_ids_to_tokens([1939])
    no_id = tokenizer.vocab['▁No']
    yes_id = tokenizer.vocab['▁Yes']  # tokenizer.convert_ids_to_tokens([3869])
    probs = torch.softmax(scores[0], dim=-1)
    # For reference, this reproduces the final output as input to the softmax:
    # model.lm_head(hidden_states[0][-1][0][-1, :])

    # The embeddings are as follows:
    # average of all (across tokens) final hidden states ::
    # final token hidden state (here this corresponds to the hidden state taken as input to the linear layer
    # that determines the vocabulary probabilities)
    embedding = torch.cat([
        torch.mean(hidden_states[0][-1][0], dim=0).unsqueeze(0),
        hidden_states[0][-1][0][-1, :].unsqueeze(0)
    ], dim=-1).to(torch.float32)
    embedding = [float(x) for x in embedding[0].cpu().numpy().tolist()]
    assert len(embedding) == EXPECTED_EMBEDDING_SIZE
    # The attributes are as follows:
    # no_prob :: yes_prob
    attributes = torch.cat([probs[0:1, no_id].unsqueeze(0),
                            probs[0:1, yes_id].unsqueeze(0)], dim=-1)
    attributes = [float(x) for x in attributes[0].cpu().numpy().tolist()]
    assert len(attributes) == 2
    logits = torch.cat([scores[0][0:1, no_id].unsqueeze(0),
                        scores[0][0:1, yes_id].unsqueeze(0)], dim=-1)
    # We also save the raw logits which we will use for comparisons to post-hoc classification methods. (E.g.,
    # for temperature scaling, we need the raw logits before the softmax.)
    logits = [float(x) for x in logits[0].cpu().numpy().tolist()]
    assert len(logits) == 2
    llm_classification = probs[0:1, no_id] < probs[0:1, yes_id]
    return embedding, attributes, logits, int(llm_classification.item())


def get_prompt(document, dataset):
    if dataset == FACTCHECK_DATA:
        return f"Here is a statement that may contain errors. <statement> {document} </statement> Is the statement true? Answer Yes if the statement is true. Answer No if the statement is false. Start your response with Yes or No."
    elif dataset == SENTIMENT_DATA:
        return f"Here is a movie review. <review> {document} </review> Is the sentiment of the movie review positive? Answer Yes if the sentiment is positive. Answer No if the sentiment is negative. Start your response with Yes or No."
    else:
        assert False


def get_existing_ids(filepath_with_name):
    existing_ids = set()
    with codecs.open(filepath_with_name, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            json_obj = json.loads(line)
            existing_ids.add(json_obj[REEXPRESS_ID_KEY])
    return existing_ids


def construct_embedding_streaming(options, model, tokenizer):
    count_incomplete_responses = 0
    output_file = options.output_file
    if Path(output_file).exists():
        existing_ids = get_existing_ids(output_file)
    else:
        existing_ids = set()

    acc = []
    acc_by_class = {}
    acc_by_predicted_class = {}
    for class_i in range(options.class_size):
        acc_by_class[class_i] = []
        acc_by_predicted_class[class_i] = []

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
            # try:
            if True:
                prompt = get_prompt(json_obj[REEXPRESS_DOCUMENT_KEY], options.dataset)
                embedding, attributes, logits, llm_classification = \
                    get_agreement_model_embedding(model=model, tokenizer=tokenizer,
                                                  document_text=prompt, device=options.main_device)
            else: #except:  # In principle, this case should never occur with this model.
                print(f"WARNING: The LLM forward pass threw an error.")
                prompt = ""
                embedding = [0.0] * EXPECTED_EMBEDDING_SIZE
                attributes = [0.0] * options.class_size
                logits = [0.0] * options.class_size
                llm_classification = 0  # default to 0
                count_incomplete_responses += 1
            new_json_obj = {}
            new_json_obj[REEXPRESS_ID_KEY] = json_obj[REEXPRESS_ID_KEY]
            new_json_obj[REEXPRESS_LABEL_KEY] = json_obj[REEXPRESS_LABEL_KEY]
            new_json_obj[REEXPRESS_DOCUMENT_KEY] = json_obj[REEXPRESS_DOCUMENT_KEY]
            new_json_obj[REEXPRESS_EMBEDDING_KEY] = embedding
            new_json_obj[REEXPRESS_ATTRIBUTES_KEY] = attributes
            new_json_obj["prompt"] = prompt
            new_json_obj["logits"] = logits
            # Note that the softmax here is different than the 'logits' field after a softmax operation,
            # since the softmax has been normalized by the full vocab. We re-save here to emphasize this distinction:
            new_json_obj["softmax"] = attributes  # same as attributes
            new_json_obj["llm_classification"] = llm_classification
            acc.append(llm_classification == new_json_obj[REEXPRESS_LABEL_KEY])
            acc_by_class[new_json_obj[REEXPRESS_LABEL_KEY]].append(
                llm_classification == new_json_obj[REEXPRESS_LABEL_KEY])
            acc_by_predicted_class[llm_classification].append(
                llm_classification == new_json_obj[REEXPRESS_LABEL_KEY])

            data_utils.save_by_appending_json_lines(output_file, [new_json_obj])
            existing_ids.add(json_obj[REEXPRESS_ID_KEY])

    print(f"Count of documents with embedding set to 0's: {count_incomplete_responses}")
    model_label = "Phi-3.5-instruct"
    print_summary(f"{model_label} accuracy", acc, total=len(acc))
    print(f"Class-conditional accuracy (i.e., stratified by TRUE class):")
    for class_i in range(options.class_size):
        print_summary(f"{model_label} accuracy true class {class_i}",
                      acc_by_class[class_i], total=len(acc))
    print(f"Prediction-conditional accuracy (i.e., stratified by PREDICTED class):")
    for class_i in range(options.class_size):
        print_summary(f"{model_label} accuracy predicted class {class_i}",
                      acc_by_predicted_class[class_i], total=len(acc))


def get_model(options):
    model_path = "microsoft/Phi-3.5-mini-instruct"
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map=options.main_device,
        torch_dtype=torch.bfloat16,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_path
    )
    set_seed(42)
    return model, tokenizer


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="-----[Add embedding data to JSON objects]-----")
    parser.add_argument("--input_file", default="", help="")
    parser.add_argument("--class_size", default=2, type=int, help="class_size")
    parser.add_argument("--dataset", default="", help="")
    parser.add_argument("--main_device", default="cpu", help="")
    parser.add_argument("--output_file", default="", help="")

    options = parser.parse_args()

    assert options.dataset in [FACTCHECK_DATA, SENTIMENT_DATA]

    start_time = time.time()
    model, tokenizer = get_model(options)
    construct_embedding_streaming(options, model, tokenizer)
    cumulative_time = time.time() - start_time
    print(f"Cumulative running time: {cumulative_time}")
