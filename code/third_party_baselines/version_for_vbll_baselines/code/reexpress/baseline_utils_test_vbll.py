# Copyright Reexpress AI, Inc. All rights reserved.
import copy

import torch

import numpy as np
import constants
import utils_model
import utils_preprocess
import data_validator

from vbll_utils_latex import print_latex_row, init_latex_rows_dict, get_acc_prop_tuple, MARGINAL_ACC_KEY, \
    MARGINAL_ADMITTED_KEY, CLASS_CONDITIONAL_ACC_KEY, CLASS_CONDITIONAL_ADMITTED_KEY, PREDICTION_CONDITIONAL_ACC_KEY, \
    PREDICTION_CONDITIONAL_ADMITTED_KEY


def get_bin(x_val_in_01, divisions=10):
    return int(np.floor(min(0.99, x_val_in_01) * divisions) % divisions)


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


def test(options, main_device):
    # This is similar in spirit to the test function of the SDM estimators, but we separate this out since this case
    # has simplified data structures. Here we have kept the variable names, but here 'sdm' is 'predictive.probs'.
    # Also, in the output, FYI that "logits" and "softmax" point to the same 'predictive.probs' value, which we
    # simply duplicate here in order to use our existing evaluation scripts.
    import baseline_utils_model_vbll
    model = \
        baseline_utils_model_vbll.load_baseline_model_torch(options.model_dir, main_device,
                                                            load_for_inference=True,
                                                            is_discriminative_vbll_model=
                                                            options.is_discriminative_vbll_model)

    alpha_prime = options.alpha
    print(f"alpha'={alpha_prime}")
    test_meta_data, _ = \
        utils_preprocess.get_metadata_lines(options, options.input_eval_set_file,
                                            reduce=False,
                                            use_embeddings=options.use_embeddings,
                                            concat_embeddings_to_attributes=options.concat_embeddings_to_attributes,
                                            calculate_summary_stats=False, is_training=False)
    test_embeddings = test_meta_data["embeddings"]
    test_labels = torch.tensor(test_meta_data["labels"])
    assert test_embeddings.shape[0] == test_labels.shape[0]
    print(f"test_embeddings.shape: {test_embeddings.shape}")
    test_set_size = test_labels.shape[0]

    latex_rows_dict_no_reject = init_latex_rows_dict(numberOfClasses=options.class_size)
    latex_rows_dict_vbll = init_latex_rows_dict(numberOfClasses=options.class_size)

    marginal_accuracy = []
    marginal_accuracy_filtered__sdm_by_alpha_prime = []

    class_conditional_accuracy = {}
    class_conditional_accuracy_filtered__sdm_by_alpha_prime = {}

    prediction_conditional_accuracy = {}
    prediction_conditional_accuracy_filtered__sdm_by_alpha_prime = {}
    for label in range(options.class_size):
        class_conditional_accuracy[label] = []
        class_conditional_accuracy_filtered__sdm_by_alpha_prime[label] = []

        prediction_conditional_accuracy[label] = []
        prediction_conditional_accuracy_filtered__sdm_by_alpha_prime[label] = []

    all_predictions_json_lines = []
    number_of_divisions = 20
    predicted_f_binned = [x for x in range(number_of_divisions)]
    true_frequency_binned = [[] for x in range(number_of_divisions)]

    true_frequency_binned_prediction_conditional = {}
    true_frequency_binned_prediction_conditional__average_sample_sizes = {}
    true_frequency_binned_class_conditional = {}
    for label in range(options.class_size):
        true_frequency_binned_prediction_conditional[label] = [[] for x in range(number_of_divisions)]
        true_frequency_binned_prediction_conditional__average_sample_sizes[label] = \
            [[] for x in range(number_of_divisions)]
        true_frequency_binned_class_conditional[label] = [[] for x in range(number_of_divisions)]
    instance_i = -1
    number_of_unlabeled_labels = 0
    number_of_ood_labels = 0

    for test_embedding, test_label in zip(test_embeddings, test_labels):
        instance_i += 1
        if instance_i % 50000 == 0:
            print(f"Currently processing index {instance_i}")
        true_test_label = test_label.item()
        assert data_validator.isValidLabel(label=true_test_label, numberOfClasses=options.class_size)
        if not data_validator.isKnownValidLabel(label=true_test_label, numberOfClasses=options.class_size):
            if true_test_label == data_validator.unlabeledLabel:
                number_of_unlabeled_labels += 1
            elif true_test_label == data_validator.oodLabel:
                number_of_ood_labels += 1
            continue
        with torch.no_grad():
            out = \
                model(test_embedding.unsqueeze(0).to(main_device))
            # There is also an out.ood_scores property, but it is just max(probs).
            probs = out.predictive.probs
        logits_output = probs.detach().cpu().squeeze()
        softmax_output = probs.detach().cpu().squeeze()
        predicted_class = torch.argmax(logits_output, dim=-1).item()

        prediction_conditional_distribution__sdm = \
            softmax_output  # note the variable name overloading; this is the vbll output

        prediction_conditional_estimate_of_predicted_class__sdm = \
            prediction_conditional_distribution__sdm[predicted_class].item()

        marginal_accuracy.append(predicted_class == true_test_label)
        class_conditional_accuracy[true_test_label].append(predicted_class == true_test_label)
        prediction_conditional_accuracy[predicted_class].append(predicted_class == true_test_label)

        json_obj = {}
        if options.prediction_output_file != "":
            json_obj["id"] = test_meta_data['uuids'][instance_i]
            json_obj["document"] = test_meta_data['lines'][instance_i]
            json_obj["label"] = true_test_label

            json_obj["logits"] = \
                logits_output.detach().cpu().numpy().tolist()
            json_obj["softmax"] = \
                softmax_output.detach().cpu().numpy().tolist()
            json_obj["prediction"] = \
                predicted_class
            all_predictions_json_lines.append(json_obj)

        if prediction_conditional_estimate_of_predicted_class__sdm >= alpha_prime:
            class_conditional_accuracy_filtered__sdm_by_alpha_prime[true_test_label].append(
                predicted_class == true_test_label)
            prediction_conditional_accuracy_filtered__sdm_by_alpha_prime[predicted_class].append(
                predicted_class == true_test_label)
            marginal_accuracy_filtered__sdm_by_alpha_prime.append(predicted_class == true_test_label)

        prediction_conditional_estimate_binned = \
            get_bin(prediction_conditional_estimate_of_predicted_class__sdm, divisions=number_of_divisions)
        true_frequency_binned[prediction_conditional_estimate_binned].append(predicted_class == true_test_label)
        true_frequency_binned_prediction_conditional[predicted_class][prediction_conditional_estimate_binned].append(
            predicted_class == true_test_label)
        true_frequency_binned_class_conditional[true_test_label][prediction_conditional_estimate_binned].append(
            predicted_class == true_test_label)

    print(f"######## Conditional estimates ########")
    for label in range(options.class_size):
        print(f"Label {label} ---")
        print_summary(f"Class-conditional accuracy: Label {label}",
                      class_conditional_accuracy[label], total=test_set_size)
        latex_rows_dict_no_reject[f"{CLASS_CONDITIONAL_ACC_KEY}{label}"], \
            latex_rows_dict_no_reject[f"{CLASS_CONDITIONAL_ADMITTED_KEY}{label}"] = \
            get_acc_prop_tuple(class_conditional_accuracy[label], total=test_set_size)

        print_summary(f"\t>>Class-conditional vbll_predicted >= {alpha_prime} accuracy: \t\tLabel {label}",
                      class_conditional_accuracy_filtered__sdm_by_alpha_prime[label], total=test_set_size)
        latex_rows_dict_vbll[f"{CLASS_CONDITIONAL_ACC_KEY}{label}"], \
            latex_rows_dict_vbll[f"{CLASS_CONDITIONAL_ADMITTED_KEY}{label}"] = \
            get_acc_prop_tuple(class_conditional_accuracy_filtered__sdm_by_alpha_prime[label],
                               total=test_set_size)

        print_summary(f"Prediction-conditional accuracy: Label {label}",
                      prediction_conditional_accuracy[label], total=test_set_size)
        latex_rows_dict_no_reject[f"{PREDICTION_CONDITIONAL_ACC_KEY}{label}"], \
            latex_rows_dict_no_reject[f"{PREDICTION_CONDITIONAL_ADMITTED_KEY}{label}"] = \
            get_acc_prop_tuple(prediction_conditional_accuracy[label], total=test_set_size)

        print_summary(f"\t>>Prediction-conditional vbll_predicted >= {alpha_prime} accuracy: "
                      f"\t\tLabel {label}",
                      prediction_conditional_accuracy_filtered__sdm_by_alpha_prime[label], total=test_set_size)
        latex_rows_dict_vbll[f"{PREDICTION_CONDITIONAL_ACC_KEY}{label}"], \
            latex_rows_dict_vbll[f"{PREDICTION_CONDITIONAL_ADMITTED_KEY}{label}"] = \
            get_acc_prop_tuple(prediction_conditional_accuracy_filtered__sdm_by_alpha_prime[label],
                               total=test_set_size)

    print(f"######## Stratified by probability ########")
    for bin in predicted_f_binned:
        print_summary(f"{bin/number_of_divisions}-{(min(number_of_divisions, bin+1))/number_of_divisions}: "
                      f"PREDICTION CONDITIONAL: Marginal",
                      true_frequency_binned[bin])
        for label in range(options.class_size):
            print(
                f"\tLabel {label} PREDICTION CONDITIONAL: "
                f"{np.mean(true_frequency_binned_prediction_conditional[label][bin])}, "
                f"out of {len(true_frequency_binned_prediction_conditional[label][bin])} || "
                f"mean sample size: "
                f"{np.mean(true_frequency_binned_prediction_conditional__average_sample_sizes[label][bin])} || "
                f"median sample size: "
                f"{np.median(true_frequency_binned_prediction_conditional__average_sample_sizes[label][bin])}")
            print(
                f"\tLabel {label} -class- -conditional-: "
                f"{np.mean(true_frequency_binned_class_conditional[label][bin])}, "
                f"out of {len(true_frequency_binned_class_conditional[label][bin])}")

    print(f"######## Marginal estimates ########")
    print(f"Marginal accuracy: {np.mean(marginal_accuracy)} out of {len(marginal_accuracy)}")
    latex_rows_dict_no_reject[MARGINAL_ACC_KEY], \
        latex_rows_dict_no_reject[MARGINAL_ADMITTED_KEY] = \
        get_acc_prop_tuple(marginal_accuracy, total=len(marginal_accuracy))

    if len(marginal_accuracy) > 0:  # it could be 0 if the eval file only includes OOD or unlabeled
        print(
            f"Filtered marginal (constrained to vbll_predicted >= {alpha_prime}): "
            f"{np.mean(marginal_accuracy_filtered__sdm_by_alpha_prime)} out of "
            f"{len(marginal_accuracy_filtered__sdm_by_alpha_prime)} "
            f"({len(marginal_accuracy_filtered__sdm_by_alpha_prime)/len(marginal_accuracy)})")
        latex_rows_dict_vbll[MARGINAL_ACC_KEY], \
            latex_rows_dict_vbll[MARGINAL_ADMITTED_KEY] = \
            get_acc_prop_tuple(marginal_accuracy_filtered__sdm_by_alpha_prime, total=len(marginal_accuracy))

    print(f"######## OOD/Unlabeled stats ########")
    print(f"Count of unlabeled labeled (i.e., label=={data_validator.unlabeledLabel}) "
          f"instances (ignored above): {number_of_unlabeled_labels} out of {test_set_size}")
    print(f"Count of OOD labeled (i.e., label=={data_validator.oodLabel}) "
          f"instances (ignored above): {number_of_ood_labels} out of {test_set_size}")
    print(f"######## ########")

    if options.prediction_output_file != "" and len(all_predictions_json_lines) > 0:
        utils_model.save_json_lines(options.prediction_output_file, all_predictions_json_lines)
        print(f">The prediction for each document (total: {len(all_predictions_json_lines)}) has been saved to "
              f"{options.prediction_output_file}")

    assert test_set_size == instance_i + 1, "ERROR: The index is mismatched."
    print_latex_row(options, alpha_prime,
                    latex_rows_dict_no_reject=latex_rows_dict_no_reject,
                    latex_rows_dict_vbll=latex_rows_dict_vbll)
