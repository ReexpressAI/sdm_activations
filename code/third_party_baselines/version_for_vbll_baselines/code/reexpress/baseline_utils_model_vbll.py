# Copyright Reexpress AI, Inc. All rights reserved.

from vbll_model import DiscVBLLMLP
from vbll_model import GenVBLLMLP
import baseline_utils_train_main_vbll
import constants

import torch
import json
import codecs
from os import path


def read_json_file(filename_with_path):
    try:
        with open(filename_with_path, 'r') as f:
            json_obj = json.load(f)
        return json_obj
    except:
        return None


def read_jsons_lines_file(filename_with_path):
    json_list = []
    with codecs.open(filename_with_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            json_obj = json.loads(line)
            json_list.append(json_obj)
    return json_list


def save_by_appending_json_lines(filename_with_path, json_list):
    with codecs.open(filename_with_path, "a", encoding="utf-8") as f:
        for json_dict in json_list:
            f.write(json.dumps(json_dict, ensure_ascii=True) + "\n")


def save_json_lines(filename_with_path, json_list):
    with codecs.open(filename_with_path, "w", encoding="utf-8") as f:
        for json_dict in json_list:
            f.write(json.dumps(json_dict, ensure_ascii=True) + "\n")


def save_lines(filename_with_path, list_of_strings_with_newlines):
    with codecs.open(filename_with_path, "w", encoding="utf-8") as f:
        f.writelines(list_of_strings_with_newlines)


def normalize_embeddings(embeddings, summary_stats):
    # return (embeddings - summary_stats[constants.STORAGE_KEY_SUMMARY_STATS_EMBEDDINGS_training_embedding_mean]) / \
    #     summary_stats[constants.STORAGE_KEY_SUMMARY_STATS_EMBEDDINGS_training_embedding_std]
    return SimilarityDistanceMagnitudeCalibrator.normalize_embeddings(
        embeddings, summary_stats)


def get_embedding_summary_stats(embeddings, is_training):
    assert is_training, f"ERROR: This must be the training/support set."
    print(f">>Collecting training set embeddings summary stats<<")
    training_embedding_mean = torch.mean(embeddings).item()
    training_embedding_std = torch.std(embeddings, correction=1).item()

    summary_stats = {
        constants.STORAGE_KEY_SUMMARY_STATS_EMBEDDINGS_training_embedding_mean: training_embedding_mean,
        constants.STORAGE_KEY_SUMMARY_STATS_EMBEDDINGS_training_embedding_std: training_embedding_std
    }
    return summary_stats


def save_baseline_model(model, model_dir, optimizer=None):  #, retain_support_index_after_archiving=True):
    model.support_index = None  # set to None to avoid saving in main weights file
    save_baseline_uncertainty_metadata(model, model_dir)
    model_statedict_output_file = path.join(model_dir, constants.FILENAME_LOCALIZER)
    torch.save(model.state_dict(), model_statedict_output_file)


def load_baseline_model_torch(model_dir, main_device, load_for_inference=False, is_discriminative_vbll_model=True):
    # There is no support index in this case.
    try:
        model_statedict_output_file = path.join(model_dir, constants.FILENAME_LOCALIZER)
        model_config, training_embedding_summary_stats = load_uncertainty_statistics_from_disk_for_baseline(
            model_dir,
            load_for_inference=load_for_inference)

        if is_discriminative_vbll_model:
            print("Loading DiscVBLLMLP() model")
            model = DiscVBLLMLP(model_config,
                                training_embedding_summary_stats=training_embedding_summary_stats).to(
                main_device)
        else:
            print("Loading GenVBLLMLP() model")
            model = GenVBLLMLP(model_config,
                               training_embedding_summary_stats=training_embedding_summary_stats).to(
                main_device)

        state_dict = torch.load(model_statedict_output_file, weights_only=True, map_location=torch.device("cpu"))
        model.load_state_dict(state_dict)

        model.eval()
        print(f"Model loaded successfully, set to eval() mode.")
        return model.to(main_device)
    except:
        print(f"ERROR: The model file is missing or incomplete. Exiting.")
        exit()


def save_baseline_uncertainty_metadata(model, model_dir):
    # build archive as json object
    json_dict = model.export_properties_to_dict()
    with codecs.open(path.join(model_dir, constants.FILENAME_UNCERTAINTY_STATISTICS), "w", encoding="utf-8") as f:
        f.write(json.dumps(json_dict, ensure_ascii=True))


def load_uncertainty_statistics_from_disk_for_baseline(model_dir, load_for_inference=False):

    json_dict = {}
    with codecs.open(path.join(model_dir, constants.FILENAME_UNCERTAINTY_STATISTICS), encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            json_dict = json.loads(line)

    model_config = baseline_utils_train_main_vbll.cfg()

    model_config.IN_FEATURES = json_dict["IN_FEATURES"]
    model_config.HIDDEN_FEATURES = json_dict["HIDDEN_FEATURES"]
    model_config.OUT_FEATURES = json_dict["OUT_FEATURES"]
    model_config.NUM_LAYERS = json_dict["NUM_LAYERS"]
    model_config.REG_WEIGHT = json_dict["REG_WEIGHT"]
    model_config.PARAM = json_dict["PARAM"]
    model_config.RETURN_OOD = json_dict["RETURN_OOD"]
    model_config.PRIOR_SCALE = json_dict["PRIOR_SCALE"]

    training_embedding_summary_stats = \
                        json_dict[constants.STORAGE_KEY_SUMMARY_STATS_EMBEDDINGS_training_embedding_summary_stats]
    return model_config, training_embedding_summary_stats


