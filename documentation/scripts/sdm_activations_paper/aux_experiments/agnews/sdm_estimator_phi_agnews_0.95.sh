#########################################################################################################
##################### Compute
#########################################################################################################

# This was run on an NVIDIA L4 GPU. Training took approximately 24 hours. That is across a relatively large
# number of epochs: 5000 total (10 iterations, each of 500 epochs).

#########################################################################################################
##################### Data
#########################################################################################################

#The data is available at https://huggingface.co/datasets/fancyzhx/ag_news.
#The approach for preprocessing is provided in agnews_phi3.5_data.sh.

#
#label: a classification label, with possible values including World (0), Sports (1), Business (2), Sci/Tech (3).
#

#########################################################################################################
##################### AGNews train and eval
#########################################################################################################


cd code/reexpress # Update with the applicable path
conda activate re_mcp_v200


RUN_SUFFIX_ID="phi_3_5_instruct"
MODEL_TYPE="classifier_r2"

DATA_DIR="/home/jupyter/data/classification/ag_news_phi35" # Update with the applicable path

# 'embedding' field is from Phi-3.5 decoder
MODEL_LABEL="phi35"
TRAIN_FILE="${DATA_DIR}/train.ag_news.${MODEL_LABEL}.jsonl"
CALIBRATION_FILE="${DATA_DIR}/calibration.ag_news.${MODEL_LABEL}.jsonl"
EVAL_FILE="${DATA_DIR}/test.ag_news.${MODEL_LABEL}.jsonl"


ALPHA=0.95
EXEMPLAR_DIMENSION=1000

MODEL_OUTPUT_DIR=/home/jupyter/models/sdm_paper/release_version/ag_news/"${RUN_SUFFIX_ID}_${MODEL_TYPE}_${ALPHA}_${EXEMPLAR_DIMENSION}"/  # Update with the applicable path

mkdir -p "${MODEL_OUTPUT_DIR}"


LEARNING_RATE=0.00001

# Note the larger number of epochs and 4 classes

python -u reexpress.py \
--input_training_set_file "${TRAIN_FILE}" \
--input_calibration_set_file "${CALIBRATION_FILE}" \
--input_eval_set_file "${EVAL_FILE}" \
--alpha=${ALPHA} \
--class_size 4 \
--seed_value 0 \
--epoch 500 \
--batch_size 50 \
--eval_batch_size 50 \
--learning_rate ${LEARNING_RATE} \
--model_dir "${MODEL_OUTPUT_DIR}" \
--number_of_random_shuffles 10 \
--maxQAvailableFromIndexer 2048 \
--exemplar_vector_dimension ${EXEMPLAR_DIMENSION} \
--main_device="cuda:0" \
--use_embeddings > ${MODEL_OUTPUT_DIR}/run1.log.txt

echo ${MODEL_OUTPUT_DIR}/run1.log.txt



#########################################################################################################
##################### Analysis
#########################################################################################################

cd code/reexpress # Update with the applicable path
conda activate re_mcp_v200


RUN_SUFFIX_ID="phi_3_5_instruct"
MODEL_TYPE="classifier_r2"


ALPHA=0.95
EXEMPLAR_DIMENSION=1000
LEARNING_RATE=0.00001

MODEL_OUTPUT_DIR=/home/jupyter/models/sdm_paper/release_version/ag_news/"${RUN_SUFFIX_ID}_${MODEL_TYPE}_${ALPHA}_${EXEMPLAR_DIMENSION}"/  # Update with the applicable path


# Run each data block in turn
EVAL_LABEL=best_iteration_data_calibration
EVAL_FILE="${MODEL_OUTPUT_DIR}/best_iteration_data/calibration.jsonl"
LATEX_DATASET_LABEL='datasetAGNewsCalibration'
LATEX_MODEL_NAME='modelPhiThreeFiveInstructSDM'

DATA_DIR="/home/jupyter/data/classification/ag_news_phi35" # Update with the applicable path
MODEL_LABEL="phi35"
EVAL_LABEL="test.ag_news"
EVAL_FILE="${DATA_DIR}/${EVAL_LABEL}.${MODEL_LABEL}.jsonl"
LATEX_DATASET_LABEL='datasetAGNews'
LATEX_MODEL_NAME='modelPhiThreeFiveInstructSDM'

MODEL_OUTPUT_DIR_WITH_SUBFOLDER=${MODEL_OUTPUT_DIR}/final_eval_output
mkdir ${MODEL_OUTPUT_DIR_WITH_SUBFOLDER}

python -u reexpress.py \
--input_training_set_file "${TRAIN_FILE}" \
--input_calibration_set_file "${CALIBRATION_FILE}" \
--input_eval_set_file "${EVAL_FILE}" \
--use_embeddings \
--alpha=${ALPHA} \
--class_size 4 \
--seed_value 0 \
--epoch 200 \
--batch_size 50 \
--eval_batch_size 50 \
--learning_rate ${LEARNING_RATE} \
--model_dir "${MODEL_OUTPUT_DIR}" \
--number_of_random_shuffles 10 \
--maxQAvailableFromIndexer 2048 \
--exemplar_vector_dimension ${EXEMPLAR_DIMENSION} \
--label_error_file=${MODEL_OUTPUT_DIR_WITH_SUBFOLDER}/"eval.${EVAL_LABEL}.possible_label_errors.jsonl" \
--predictions_in_high_reliability_region_file=${MODEL_OUTPUT_DIR_WITH_SUBFOLDER}/"eval.${EVAL_LABEL}.high_reliability.jsonl" \
--prediction_output_file=${MODEL_OUTPUT_DIR_WITH_SUBFOLDER}/"eval.${EVAL_LABEL}.all_predictions.jsonl" \
--eval_only \
--main_device="cuda:0" \
--construct_results_latex_table_rows \
--additional_latex_meta_data="${LATEX_DATASET_LABEL},${LATEX_MODEL_NAME}" > ${MODEL_OUTPUT_DIR_WITH_SUBFOLDER}/"eval.${EVAL_LABEL}.version_2.1.0.log.txt"

echo "Eval Label: ${EVAL_LABEL}"
echo "Possible label errors (sorted) file: "${MODEL_OUTPUT_DIR_WITH_SUBFOLDER}/"eval.${EVAL_LABEL}.possible_label_errors.jsonl"
echo "Valid index-conditional predictions (sorted) file: "${MODEL_OUTPUT_DIR_WITH_SUBFOLDER}/"eval.${EVAL_LABEL}.high_reliability.jsonl"
echo "All predictions file: "${MODEL_OUTPUT_DIR_WITH_SUBFOLDER}/"eval.${EVAL_LABEL}.all_predictions.jsonl"
echo "Eval log file: "${MODEL_OUTPUT_DIR_WITH_SUBFOLDER}/"eval.${EVAL_LABEL}.version_2.1.0.log.txt"

#########################################################################################################
##################### analysis
#########################################################################################################

cd code/reexpress # Update with the applicable path
conda activate re_mcp_v200


RUN_SUFFIX_ID="phi_3_5_instruct"
MODEL_TYPE="classifier_r2"


ALPHA=0.95
EXEMPLAR_DIMENSION=1000
LEARNING_RATE=0.00001

MODEL_OUTPUT_DIR=/home/jupyter/models/sdm_paper/release_version/ag_news/"${RUN_SUFFIX_ID}_${MODEL_TYPE}_${ALPHA}_${EXEMPLAR_DIMENSION}"/  # Update with the applicable path

INPUT_DIR="${MODEL_OUTPUT_DIR}/final_eval_output"

# Run each data block in turn
INPUT_FILE=${INPUT_DIR}/eval.best_iteration_data_calibration.all_predictions.jsonl
FILE_LABEL="Calibration (not held-out)"
OUTPUT_FILE_PREFIX="Calibration"
X_BIN_WIDTH=100

# primary test set
INPUT_FILE=${INPUT_DIR}/eval.ood_eval.all_predictions.jsonl
FILE_LABEL="AGNews test-set"
OUTPUT_FILE_PREFIX="AGNews-test"
X_BIN_WIDTH=100

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
--model_version_label="v2.1.0" \
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
--model_version_label="v2.1.0" \
--save_file_prefix=${OUTPUT_DIR}/${OUTPUT_FILE_PREFIX}

#########################################################################################################
##################### Analysis -- Ensemble eval :: For reference, here is an example of running the
##################### evaluation script introduced in v2.3.0 of the code that has an option for
##################### constructing an ensemble across the models from each of the J=10 training iterations.
#########################################################################################################

cd code/reexpress # Update with the applicable path (v2.3.0)
conda activate re_mcp_v200


RUN_SUFFIX_ID="phi_3_5_instruct"
MODEL_TYPE="classifier_r2"


ALPHA=0.95
EXEMPLAR_DIMENSION=1000
LEARNING_RATE=0.00001

MODEL_OUTPUT_DIR=/home/jupyter/models/sdm_paper/release_version/ag_news/"${RUN_SUFFIX_ID}_${MODEL_TYPE}_${ALPHA}_${EXEMPLAR_DIMENSION}"/  # Update with the applicable path


DATA_DIR="/home/jupyter/data/classification/ag_news_phi35"  # Update with the applicable path
MODEL_LABEL="phi35"
LATEX_MODEL_NAME='modelPhiThreeFiveInstructSDM'

MODEL_OUTPUT_DIR_WITH_SUBFOLDER=${MODEL_OUTPUT_DIR}/final_eval_output_models_0_through_9
mkdir ${MODEL_OUTPUT_DIR_WITH_SUBFOLDER}

STANDARD_OUT_LOG_FILE=${MODEL_OUTPUT_DIR_WITH_SUBFOLDER}/eval_standard_out.log.txt
echo ${STANDARD_OUT_LOG_FILE}

for EVAL_LABEL in "test.ag_news" "best_iteration_data_calibration"; do

# Set LATEX_DATASET_LABEL based on EVAL_LABEL
if [ "$EVAL_LABEL" = "best_iteration_data_calibration" ]; then
    # Calibration (i.e., the original input to --input_calibration_set_file) is shuffled during training, so we retrieve the final shuffle associated with this model iteration
    EVAL_FILE="${MODEL_OUTPUT_DIR}/best_iteration_data/calibration.jsonl"
    LATEX_DATASET_LABEL="datasetAGNewsCalibration"
else
    EVAL_FILE="${DATA_DIR}/${EVAL_LABEL}.${MODEL_LABEL}.jsonl"
fi
if [ "$EVAL_LABEL" = "test.ag_news" ]; then
    LATEX_DATASET_LABEL="datasetAGNews"
fi

python -u reexpress.py \
--input_training_set_file "${TRAIN_FILE}" \
--input_calibration_set_file "${CALIBRATION_FILE}" \
--input_eval_set_file "${EVAL_FILE}" \
--use_embeddings \
--alpha=${ALPHA} \
--class_size 4 \
--seed_value 0 \
--epoch 200 \
--batch_size 50 \
--eval_batch_size 50 \
--learning_rate ${LEARNING_RATE} \
--model_dir "${MODEL_OUTPUT_DIR}" \
--number_of_random_shuffles 10 \
--maxQAvailableFromIndexer 2048 \
--exemplar_vector_dimension ${EXEMPLAR_DIMENSION} \
--main_device="cuda:0" \
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
--additional_latex_meta_data="${LATEX_DATASET_LABEL},${LATEX_MODEL_NAME}" > ${MODEL_OUTPUT_DIR_WITH_SUBFOLDER}/"eval.${EVAL_LABEL}.version_2.3.0.log.txt"

echo "Eval Label: ${EVAL_LABEL}" >> ${STANDARD_OUT_LOG_FILE}
echo "Possible label errors in HR_LOWER region (sorted) file: "${MODEL_OUTPUT_DIR_WITH_SUBFOLDER}/"eval.${EVAL_LABEL}.possible_label_errors.hr_lower.jsonl" >> ${STANDARD_OUT_LOG_FILE}
echo "High reliablity region LOWER predictions (sorted) file: "${MODEL_OUTPUT_DIR_WITH_SUBFOLDER}/"eval.${EVAL_LABEL}.high_reliability_lower.jsonl" >> ${STANDARD_OUT_LOG_FILE}
echo "" >> ${STANDARD_OUT_LOG_FILE}
echo "Possible label errors in HR region (sorted) file: "${MODEL_OUTPUT_DIR_WITH_SUBFOLDER}/"eval.${EVAL_LABEL}.possible_label_errors.jsonl" >> ${STANDARD_OUT_LOG_FILE}
echo "High reliablity region predictions (sorted) file: "${MODEL_OUTPUT_DIR_WITH_SUBFOLDER}/"eval.${EVAL_LABEL}.high_reliability.jsonl" >> ${STANDARD_OUT_LOG_FILE}
echo "" >> ${STANDARD_OUT_LOG_FILE}
echo "All predictions file: "${MODEL_OUTPUT_DIR_WITH_SUBFOLDER}/"eval.${EVAL_LABEL}.all_predictions.jsonl" >> ${STANDARD_OUT_LOG_FILE}
echo "-----Ensemble-----" >> ${STANDARD_OUT_LOG_FILE}
echo "ENSEMBLE Possible label errors in HR_LOWER region (sorted) file: >> ${STANDARD_OUT_LOG_FILE} "${MODEL_OUTPUT_DIR_WITH_SUBFOLDER}/"eval.${EVAL_LABEL}.possible_label_errors.hr_lower.ensemble.jsonl" >> ${STANDARD_OUT_LOG_FILE}
echo "ENSEMBLE High reliablity region LOWER predictions (sorted) file: "${MODEL_OUTPUT_DIR_WITH_SUBFOLDER}/"eval.${EVAL_LABEL}.high_reliability_lower.ensemble.jsonl" >> ${STANDARD_OUT_LOG_FILE}
echo "" >> ${STANDARD_OUT_LOG_FILE}
echo "ENSEMBLE Possible label errors in HR region (sorted) file: "${MODEL_OUTPUT_DIR_WITH_SUBFOLDER}/"eval.${EVAL_LABEL}.possible_label_errors.ensemble.jsonl" >> ${STANDARD_OUT_LOG_FILE}
echo "ENSEMBLE High reliablity region predictions (sorted) file: "${MODEL_OUTPUT_DIR_WITH_SUBFOLDER}/"eval.${EVAL_LABEL}.high_reliability.ensemble.jsonl" >> ${STANDARD_OUT_LOG_FILE}
echo "" >> ${STANDARD_OUT_LOG_FILE}
echo "ENSEMBLE All predictions file: "${MODEL_OUTPUT_DIR_WITH_SUBFOLDER}/"eval.${EVAL_LABEL}.all_predictions.ensemble.jsonl" >> ${STANDARD_OUT_LOG_FILE}
echo "" >> ${STANDARD_OUT_LOG_FILE}
echo "Eval log file: "${MODEL_OUTPUT_DIR_WITH_SUBFOLDER}/"eval.${EVAL_LABEL}.version_2.3.0.log.txt" >> ${STANDARD_OUT_LOG_FILE}
echo "" >> ${STANDARD_OUT_LOG_FILE}
echo "" >> ${STANDARD_OUT_LOG_FILE}

done
