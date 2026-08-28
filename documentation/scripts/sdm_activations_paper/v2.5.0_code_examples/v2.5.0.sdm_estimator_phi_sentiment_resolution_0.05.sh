#########################################################################################################
##################### Changes introduced in version 2.5.0
#########################################################################################################

## The following assumes release version 2.5.0.

### Note that unlike previous versions, 'alpha' is now described via the offset resolution. For example,
### if the target alpha of the most conservative region is 0.95, set --alpha_resolution=0.05. If such
### an alpha is not obtainable, the calibration algorithm will automatically search for the next highest
### available region.
###
### This change from --alpha=${ALPHA} to --alpha_resolution=${ALPHA_RESOLUTION} is the main command-line
### change from earlier versions.
###
###
### v2.5.0 has a new option to convert the json lines files to an on-disk format.
### You can still use the raw json lines files for convenience, but the
### on-disk format has a lower memory footprint. Additionally, when using this format,
### only the indexes of the particular shuffle of D_ca is saved for the chosen model iteration,
### rather than the full file. To reevaluate on that calibration split, use the flag
### --eval_on_best_iteration_calibration_split with reexpress.py. Here we use the JSON lines format,
### but see reexpress_mcp_server/documentation/model_details/release/v2.5.0/train_and_eval_sdm_estimator_v2.5.0.sh
### for an example of the new format, which requires a one-time conversion step from JSON lines files
### using aux_convert_to_reexpress_dataset.py.


#########################################################################################################
##################### Compute
#########################################################################################################

# 12 GB of GPU memory should be sufficient. (That is a conservative estimate;
# much less is likely needed given a batch size of 50.) This example runs on Apple silicon
# by using --main_device="mps". This can also be run on CPU by
# setting --main_device="cpu" or GPU using, e.g., --main_device="cuda:0". Using the new data format
# mentioned above can reduce memory usage, if needed.

#########################################################################################################
##################### Sentiment train and eval
#########################################################################################################


cd code/reexpress # Update with the applicable path
conda activate re_mcp_v250


RUN_SUFFIX_ID="phi_3_5_instruct"
MODEL_TYPE="classifier_v2.5.0"

DATA_DIR="/home/jupyter/data/classification/sentiment_phi35"  # Update with the applicable path

# 'embedding' field is from Phi-3.5 decoder
MODEL_LABEL="phi35"
TRAIN_FILE="${DATA_DIR}/training_set.${MODEL_LABEL}.jsonl"
CALIBRATION_FILE="${DATA_DIR}/calibration_set.${MODEL_LABEL}.jsonl"


EVAL_LABEL="validation_set"  # primary test set
EVAL_LABEL="eval_set"  # a small eval set
EVAL_LABEL="SemEval2017-task4-test.subtask-A.english.binaryevalformat.balanced"  # OOD test set
EVAL_FILE="${DATA_DIR}/eval_sets/${EVAL_LABEL}.${MODEL_LABEL}.jsonl"


ALPHA_RESOLUTION=0.05
EXEMPLAR_DIMENSION=1000

MODEL_OUTPUT_DIR=/home/jupyter/models/sdm_paper/release_version/sentiment/"${RUN_SUFFIX_ID}_${MODEL_TYPE}_${ALPHA_RESOLUTION}_${EXEMPLAR_DIMENSION}"/ # Update with the applicable path

mkdir -p "${MODEL_OUTPUT_DIR}"


LEARNING_RATE=0.00001

# train:

python -u reexpress.py \
--input_training_set_file "${TRAIN_FILE}" \
--input_calibration_set_file "${CALIBRATION_FILE}" \
--input_eval_set_file "${EVAL_FILE}" \
--alpha_resolution=${ALPHA_RESOLUTION} \
--class_size 2 \
--seed_value 0 \
--epoch 200 \
--batch_size 50 \
--eval_batch_size 50 \
--learning_rate ${LEARNING_RATE} \
--model_dir "${MODEL_OUTPUT_DIR}" \
--number_of_random_shuffles 10 \
--maxQAvailableFromIndexer 2048 \
--exemplar_vector_dimension ${EXEMPLAR_DIMENSION} \
--main_device="mps" \
--use_embeddings > ${MODEL_OUTPUT_DIR}/run1.log.txt

echo ${MODEL_OUTPUT_DIR}/run1.log.txt



#########################################################################################################
##################### Analysis
#########################################################################################################

cd code/reexpress # Update with the applicable path
conda activate re_mcp_v250


RUN_SUFFIX_ID="phi_3_5_instruct"
MODEL_TYPE="classifier_v2.5.0"

ALPHA_RESOLUTION=0.05
EXEMPLAR_DIMENSION=1000
LEARNING_RATE=0.00001

MODEL_OUTPUT_DIR=/home/jupyter/models/sdm_paper/release_version/sentiment/"${RUN_SUFFIX_ID}_${MODEL_TYPE}_${ALPHA_RESOLUTION}_${EXEMPLAR_DIMENSION}"/ # Update with the applicable path


DATA_DIR="/home/jupyter/data/classification/sentiment_phi35"  # Update with the applicable path
MODEL_LABEL="phi35"
LATEX_MODEL_NAME='modelPhiThreeFiveInstructSDM'

    
MODEL_OUTPUT_DIR_WITH_SUBFOLDER=${MODEL_OUTPUT_DIR}/final_eval_output
mkdir ${MODEL_OUTPUT_DIR_WITH_SUBFOLDER}

for EVAL_LABEL in "best_iteration_data_calibration" "validation_set" "eval_set" "SemEval2017-task4-test.subtask-A.english.binaryevalformat.balanced" "validation_set.ood_random_shuffle" "eval_set.ood_random_shuffle" "SemEval2017-task4-test.subtask-A.english.binaryevalformat.balanced.ood_random_shuffle"; do

if [ "$EVAL_LABEL" = "best_iteration_data_calibration" ]; then
    LATEX_DATASET_LABEL="datasetSentiment ca"
    EVAL_FILE="${MODEL_OUTPUT_DIR}/best_iteration_data/calibration.jsonl"
else
    EVAL_FILE="${DATA_DIR}/eval_sets/${EVAL_LABEL}.${MODEL_LABEL}.jsonl"
fi

# Set LATEX_DATASET_LABEL based on EVAL_LABEL
if [ "$EVAL_LABEL" = "validation_set" ]; then
    LATEX_DATASET_LABEL="datasetSentiment"
elif [ "$EVAL_LABEL" = "eval_set" ]; then
    LATEX_DATASET_LABEL="datasetSentimentSmall"
elif [ "$EVAL_LABEL" = "SemEval2017-task4-test.subtask-A.english.binaryevalformat.balanced" ]; then
    LATEX_DATASET_LABEL="datasetSentimentOOD"
elif [ "$EVAL_LABEL" = "validation_set.ood_random_shuffle" ]; then
    LATEX_DATASET_LABEL="datasetSentimentShuffled"
elif [ "$EVAL_LABEL" = "eval_set.ood_random_shuffle" ]; then
    LATEX_DATASET_LABEL="datasetSentimentSmallShuffled"
elif [ "$EVAL_LABEL" = "SemEval2017-task4-test.subtask-A.english.binaryevalformat.balanced.ood_random_shuffle" ]; then
    LATEX_DATASET_LABEL="datasetSentimentOODShuffled"
fi

python -u reexpress.py \
--input_training_set_file "${TRAIN_FILE}" \
--input_calibration_set_file "${CALIBRATION_FILE}" \
--input_eval_set_file "${EVAL_FILE}" \
--use_embeddings \
--alpha_resolution=${ALPHA_RESOLUTION} \
--class_size 2 \
--seed_value 0 \
--epoch 200 \
--batch_size 50 \
--eval_batch_size 50 \
--learning_rate ${LEARNING_RATE} \
--model_dir "${MODEL_OUTPUT_DIR}" \
--number_of_random_shuffles 10 \
--maxQAvailableFromIndexer 2048 \
--exemplar_vector_dimension ${EXEMPLAR_DIMENSION} \
--label_error_hr_lower_file=${MODEL_OUTPUT_DIR_WITH_SUBFOLDER}/"eval.${EVAL_LABEL}.possible_label_errors.hr_lower.jsonl" \
--predictions_in_high_reliability_region_lower_file=${MODEL_OUTPUT_DIR_WITH_SUBFOLDER}/"eval.${EVAL_LABEL}.high_reliability_lower.jsonl" \
--label_error_file=${MODEL_OUTPUT_DIR_WITH_SUBFOLDER}/"eval.${EVAL_LABEL}.possible_label_errors.jsonl" \
--predictions_in_high_reliability_region_file=${MODEL_OUTPUT_DIR_WITH_SUBFOLDER}/"eval.${EVAL_LABEL}.high_reliability.jsonl" \
--prediction_output_file=${MODEL_OUTPUT_DIR_WITH_SUBFOLDER}/"eval.${EVAL_LABEL}.all_predictions.jsonl" \
--eval_only \
--main_device="mps" \
--construct_results_latex_table_rows \
--additional_latex_meta_data="${LATEX_DATASET_LABEL},${LATEX_MODEL_NAME}" > ${MODEL_OUTPUT_DIR_WITH_SUBFOLDER}/"eval.${EVAL_LABEL}.version_2.5.0.log.txt"

echo "Eval Label: ${EVAL_LABEL}"
echo "Possible label errors in most conservative HR_LOWER region (sorted) file: "${MODEL_OUTPUT_DIR_WITH_SUBFOLDER}/"eval.${EVAL_LABEL}.possible_label_errors.hr_lower.jsonl"
echo "Most conservative high reliablity region LOWER predictions (sorted) file: "${MODEL_OUTPUT_DIR_WITH_SUBFOLDER}/"eval.${EVAL_LABEL}.high_reliability_lower.jsonl"
echo "Possible label errors in most conservative HR region (sorted) file: "${MODEL_OUTPUT_DIR_WITH_SUBFOLDER}/"eval.${EVAL_LABEL}.possible_label_errors.jsonl"
echo "Most conservative high reliablity region predictions (sorted) file: "${MODEL_OUTPUT_DIR_WITH_SUBFOLDER}/"eval.${EVAL_LABEL}.high_reliability.jsonl"
echo "All predictions file: "${MODEL_OUTPUT_DIR_WITH_SUBFOLDER}/"eval.${EVAL_LABEL}.all_predictions.jsonl"
echo "Eval log file: "${MODEL_OUTPUT_DIR_WITH_SUBFOLDER}/"eval.${EVAL_LABEL}.version_2.5.0.log.txt"
echo "\n\n"
done


#########################################################################################################
##################### Graph
#########################################################################################################

cd code/reexpress # Update with the applicable path
conda activate re_mcp_v250


RUN_SUFFIX_ID="phi_3_5_instruct"
MODEL_TYPE="classifier_v2.5.0"

ALPHA_RESOLUTION=0.05
EXEMPLAR_DIMENSION=1000
LEARNING_RATE=0.00001

MODEL_OUTPUT_DIR=/home/jupyter/models/sdm_paper/release_version/sentiment/"${RUN_SUFFIX_ID}_${MODEL_TYPE}_${ALPHA_RESOLUTION}_${EXEMPLAR_DIMENSION}"/ # Update with the applicable path

INPUT_DIR="${MODEL_OUTPUT_DIR}/final_eval_output"

X_BIN_WIDTH=200

# Run each data block in turn

INPUT_FILE=${INPUT_DIR}/eval.best_iteration_data_calibration.all_predictions.jsonl
FILE_LABEL="Calibration (not held-out)"
OUTPUT_FILE_PREFIX="Calibration"


# primary test set
INPUT_FILE=${INPUT_DIR}/eval.validation_set.all_predictions.jsonl
FILE_LABEL="Sentiment test-set"
OUTPUT_FILE_PREFIX="Sentiment-test"

INPUT_FILE=${INPUT_DIR}/eval.eval_set.all_predictions.jsonl
FILE_LABEL="Sentiment test-set small"
OUTPUT_FILE_PREFIX="Sentiment-test-small"


X_BIN_WIDTH=50
INPUT_FILE=${INPUT_DIR}/eval.SemEval2017-task4-test.subtask-A.english.binaryevalformat.balanced.all_predictions.jsonl
FILE_LABEL="Sentiment-OOD"
OUTPUT_FILE_PREFIX="Sentiment-OOD"


INPUT_FILE=${INPUT_DIR}/eval.validation_set.ood_random_shuffle.all_predictions.jsonl
FILE_LABEL="Sentiment test-set shuffled (OOD)"
OUTPUT_FILE_PREFIX="Sentiment-test-shuffled"

INPUT_FILE=${INPUT_DIR}/eval.eval_set.ood_random_shuffle.all_predictions.jsonl
FILE_LABEL="Sentiment test-set small shuffled (OOD)"
OUTPUT_FILE_PREFIX="Sentiment-test-small-shuffled"

INPUT_FILE=${INPUT_DIR}/eval.SemEval2017-task4-test.subtask-A.english.binaryevalformat.balanced.ood_random_shuffle.all_predictions.jsonl
FILE_LABEL="Sentiment-OOD shuffled"
OUTPUT_FILE_PREFIX="Sentiment-OOD-shuffled"


OUTPUT_DIR="${MODEL_OUTPUT_DIR}/final_eval_output/graphs"
mkdir -p ${OUTPUT_DIR}

python -u utils_graph_output.py \
--input_file="${INPUT_FILE}" \
--class_size=2 \
--model_dir "${MODEL_OUTPUT_DIR}" \
--graph_thresholds \
--data_label="${FILE_LABEL}" \
--constant_histogram_count_axis \
--x_axis_histogram_width=${X_BIN_WIDTH} \
--model_version_label="v2.5.0" \
--save_file_prefix=${OUTPUT_DIR}/${OUTPUT_FILE_PREFIX}

python -u utils_graph_output.py \
--input_file="${INPUT_FILE}" \
--class_size=2 \
--model_dir "${MODEL_OUTPUT_DIR}" \
--graph_all_points \
--graph_thresholds \
--data_label="${FILE_LABEL}" \
--constant_histogram_count_axis \
--x_axis_histogram_width=${X_BIN_WIDTH} \
--model_version_label="v2.5.0" \
--save_file_prefix=${OUTPUT_DIR}/${OUTPUT_FILE_PREFIX}

# There is now a new option to subset the points by alpha region(s). To do so, use
#   --graph_class_and_prediction_conditional_estimates_min_region and
#   --graph_class_and_prediction_conditional_estimates_max_region
# For example, the following will graph any point in a region for which the class- and prediction-conditional accuracy is estimated to be at least 0.9:

python -u utils_graph_output.py \
--input_file="${INPUT_FILE}" \
--class_size=2 \
--model_dir "${MODEL_OUTPUT_DIR}" \
--graph_thresholds \
--data_label="${FILE_LABEL}" \
--constant_histogram_count_axis \
--x_axis_histogram_width=${X_BIN_WIDTH} \
--model_version_label="v2.5.0" \
--graph_class_and_prediction_conditional_estimates_min_region=0.9 \
--graph_class_and_prediction_conditional_estimates_max_region=1.0 \
--save_file_prefix=${OUTPUT_DIR}/${OUTPUT_FILE_PREFIX}

#########################################################################################################
##################### Analysis -- Ensemble eval :: For reference, here is an example of
##################### constructing an ensemble across the models from each of the J=10 training iterations.
#########################################################################################################

cd code/reexpress # Update with the applicable path (v2.5.0)
conda activate re_mcp_v250


RUN_SUFFIX_ID="phi_3_5_instruct"
MODEL_TYPE="classifier_v2.5.0"

ALPHA_RESOLUTION=0.05
EXEMPLAR_DIMENSION=1000
LEARNING_RATE=0.00001

MODEL_OUTPUT_DIR=/home/jupyter/models/sdm_paper/release_version/sentiment/"${RUN_SUFFIX_ID}_${MODEL_TYPE}_${ALPHA_RESOLUTION}_${EXEMPLAR_DIMENSION}"/ # Update with the applicable path

DATA_DIR="/home/jupyter/data/classification/sentiment_phi35"  # Update with the applicable path
MODEL_LABEL="phi35"
LATEX_MODEL_NAME='modelPhiThreeFiveInstructSDM'

MODEL_OUTPUT_DIR_WITH_SUBFOLDER=${MODEL_OUTPUT_DIR}/final_eval_output_models_0_through_9
mkdir ${MODEL_OUTPUT_DIR_WITH_SUBFOLDER}

STANDARD_OUT_LOG_FILE=${MODEL_OUTPUT_DIR_WITH_SUBFOLDER}/eval_standard_out.log.txt
echo ${STANDARD_OUT_LOG_FILE}

for EVAL_LABEL in "validation_set" "eval_set" "SemEval2017-task4-test.subtask-A.english.binaryevalformat.balanced" "validation_set.ood_random_shuffle" "eval_set.ood_random_shuffle" "SemEval2017-task4-test.subtask-A.english.binaryevalformat.balanced.ood_random_shuffle" "best_iteration_data_calibration"; do

# Set LATEX_DATASET_LABEL based on EVAL_LABEL
if [ "$EVAL_LABEL" = "best_iteration_data_calibration" ]; then
    # Calibration (i.e., the original input to --input_calibration_set_file) is shuffled during training, so we retrieve the final shuffle associated with this model iteration
    EVAL_FILE="${MODEL_OUTPUT_DIR}/best_iteration_data/calibration.jsonl"
    LATEX_DATASET_LABEL="datasetSentiment ca"
else
    EVAL_FILE="${DATA_DIR}/eval_sets/${EVAL_LABEL}.${MODEL_LABEL}.jsonl"
fi
if [ "$EVAL_LABEL" = "validation_set" ]; then
    LATEX_DATASET_LABEL="datasetSentiment"
elif [ "$EVAL_LABEL" = "eval_set" ]; then
    LATEX_DATASET_LABEL="datasetSentimentSmall"
elif [ "$EVAL_LABEL" = "SemEval2017-task4-test.subtask-A.english.binaryevalformat.balanced" ]; then
    LATEX_DATASET_LABEL="datasetSentimentOOD"
elif [ "$EVAL_LABEL" = "validation_set.ood_random_shuffle" ]; then
    LATEX_DATASET_LABEL="datasetSentimentShuffled"
elif [ "$EVAL_LABEL" = "eval_set.ood_random_shuffle" ]; then
    LATEX_DATASET_LABEL="datasetSentimentSmallShuffled"
elif [ "$EVAL_LABEL" = "SemEval2017-task4-test.subtask-A.english.binaryevalformat.balanced.ood_random_shuffle" ]; then
    LATEX_DATASET_LABEL="datasetSentimentOODShuffled"
fi

python -u reexpress.py \
--input_training_set_file "${TRAIN_FILE}" \
--input_calibration_set_file "${CALIBRATION_FILE}" \
--input_eval_set_file "${EVAL_FILE}" \
--use_embeddings \
--alpha_resolution=${ALPHA_RESOLUTION} \
--class_size 2 \
--seed_value 0 \
--epoch 200 \
--batch_size 50 \
--eval_batch_size 50 \
--learning_rate ${LEARNING_RATE} \
--model_dir "${MODEL_OUTPUT_DIR}" \
--number_of_random_shuffles 10 \
--maxQAvailableFromIndexer 2048 \
--exemplar_vector_dimension ${EXEMPLAR_DIMENSION} \
--main_device="mps" \
--label_error_hr_lower_file=${MODEL_OUTPUT_DIR_WITH_SUBFOLDER}/"eval.${EVAL_LABEL}.possible_label_errors.hr_lower.jsonl" \
--predictions_in_high_reliability_region_lower_file=${MODEL_OUTPUT_DIR_WITH_SUBFOLDER}/"eval.${EVAL_LABEL}.high_reliability_lower.jsonl" \
--label_error_file=${MODEL_OUTPUT_DIR_WITH_SUBFOLDER}/"eval.${EVAL_LABEL}.possible_label_errors.jsonl" \
--predictions_in_high_reliability_region_file=${MODEL_OUTPUT_DIR_WITH_SUBFOLDER}/"eval.${EVAL_LABEL}.high_reliability.jsonl" \
--prediction_output_file=${MODEL_OUTPUT_DIR_WITH_SUBFOLDER}/"eval.${EVAL_LABEL}.all_predictions.jsonl" \
--eval_only \
--eval_ensemble \
--eval_ensemble_start_iteration=0 \
--eval_ensemble_end_iteration=9 \
--eval_ensemble_label_error_hr_lower_file=${MODEL_OUTPUT_DIR_WITH_SUBFOLDER}/"eval.${EVAL_LABEL}.possible_label_errors.hr_lower.ensemble.jsonl" \
--eval_ensemble_predictions_in_high_reliability_region_lower_file=${MODEL_OUTPUT_DIR_WITH_SUBFOLDER}/"eval.${EVAL_LABEL}.high_reliability_lower.ensemble.jsonl" \
--eval_ensemble_label_error_file=${MODEL_OUTPUT_DIR_WITH_SUBFOLDER}/"eval.${EVAL_LABEL}.possible_label_errors.ensemble.jsonl" \
--eval_ensemble_predictions_in_high_reliability_region_file=${MODEL_OUTPUT_DIR_WITH_SUBFOLDER}/"eval.${EVAL_LABEL}.high_reliability.ensemble.jsonl" \
--eval_ensemble_prediction_output_file=${MODEL_OUTPUT_DIR_WITH_SUBFOLDER}/"eval.${EVAL_LABEL}.all_predictions.ensemble.jsonl" \
--construct_results_latex_table_rows \
--additional_latex_meta_data="${LATEX_DATASET_LABEL},${LATEX_MODEL_NAME}" > ${MODEL_OUTPUT_DIR_WITH_SUBFOLDER}/"eval.${EVAL_LABEL}.version_2.5.0.log.txt"

echo "======================================================" >> ${STANDARD_OUT_LOG_FILE}
echo "Eval Label: ${EVAL_LABEL}" >> ${STANDARD_OUT_LOG_FILE}
echo "Possible label errors in most conservative HR_LOWER region (sorted) file: "${MODEL_OUTPUT_DIR_WITH_SUBFOLDER}/"eval.${EVAL_LABEL}.possible_label_errors.hr_lower.jsonl" >> ${STANDARD_OUT_LOG_FILE}
echo "Most conservative high reliablity region LOWER predictions (sorted) file: "${MODEL_OUTPUT_DIR_WITH_SUBFOLDER}/"eval.${EVAL_LABEL}.high_reliability_lower.jsonl" >> ${STANDARD_OUT_LOG_FILE}
echo "" >> ${STANDARD_OUT_LOG_FILE}
echo "Possible label errors in most conservative HR region (sorted) file: "${MODEL_OUTPUT_DIR_WITH_SUBFOLDER}/"eval.${EVAL_LABEL}.possible_label_errors.jsonl" >> ${STANDARD_OUT_LOG_FILE}
echo "Most conservative high reliablity region predictions (sorted) file: "${MODEL_OUTPUT_DIR_WITH_SUBFOLDER}/"eval.${EVAL_LABEL}.high_reliability.jsonl" >> ${STANDARD_OUT_LOG_FILE}
echo "" >> ${STANDARD_OUT_LOG_FILE}
echo "All predictions file: "${MODEL_OUTPUT_DIR_WITH_SUBFOLDER}/"eval.${EVAL_LABEL}.all_predictions.jsonl" >> ${STANDARD_OUT_LOG_FILE}
echo "-----Ensemble-----" >> ${STANDARD_OUT_LOG_FILE}
echo "ENSEMBLE Possible label errors in HR_LOWER region (sorted) file:
    "${MODEL_OUTPUT_DIR_WITH_SUBFOLDER}/"eval.${EVAL_LABEL}.possible_label_errors.hr_lower.ensemble.jsonl" >> ${STANDARD_OUT_LOG_FILE}
echo "ENSEMBLE High reliablity region LOWER predictions (sorted) file: "${MODEL_OUTPUT_DIR_WITH_SUBFOLDER}/"eval.${EVAL_LABEL}.high_reliability_lower.ensemble.jsonl" >> ${STANDARD_OUT_LOG_FILE}
echo "" >> ${STANDARD_OUT_LOG_FILE}
echo "ENSEMBLE Possible label errors in HR region (sorted) file: "${MODEL_OUTPUT_DIR_WITH_SUBFOLDER}/"eval.${EVAL_LABEL}.possible_label_errors.ensemble.jsonl" >> ${STANDARD_OUT_LOG_FILE}
echo "ENSEMBLE High reliablity region predictions (sorted) file: "${MODEL_OUTPUT_DIR_WITH_SUBFOLDER}/"eval.${EVAL_LABEL}.high_reliability.ensemble.jsonl" >> ${STANDARD_OUT_LOG_FILE}
echo "" >> ${STANDARD_OUT_LOG_FILE}
echo "ENSEMBLE All predictions file: "${MODEL_OUTPUT_DIR_WITH_SUBFOLDER}/"eval.${EVAL_LABEL}.all_predictions.ensemble.jsonl" >> ${STANDARD_OUT_LOG_FILE}
echo "" >> ${STANDARD_OUT_LOG_FILE}
echo "Eval log file: "${MODEL_OUTPUT_DIR_WITH_SUBFOLDER}/"eval.${EVAL_LABEL}.version_2.5.0.log.txt" >> ${STANDARD_OUT_LOG_FILE}
echo "" >> ${STANDARD_OUT_LOG_FILE}
echo "" >> ${STANDARD_OUT_LOG_FILE}

done
