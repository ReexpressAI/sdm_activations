#########################################################################################################
##################### Compute
#########################################################################################################

# 12 GB of GPU memory should be sufficient. (That is a conservative estimate;
# much less is likely needed given a batch size of 50.) This can also be run on CPU by
# setting --main_device="cpu".

#########################################################################################################
##################### Sentiment train and eval
#########################################################################################################


cd code/reexpress # Update with the applicable path
conda activate re_mcp_v200


RUN_SUFFIX_ID="mixtral_8x7b"
MODEL_TYPE="classifier"

DATA_DIR="/home/jupyter/data/classification/sentiment_mixtral_8x7b"  # Update with the applicable path

# 'embedding' field is from mixtral_8x7b decoder
MODEL_LABEL="mixtral_8x7b"
TRAIN_FILE="${DATA_DIR}/training_set.${MODEL_LABEL}.jsonl"
CALIBRATION_FILE="${DATA_DIR}/calibration_set.${MODEL_LABEL}.jsonl"


EVAL_LABEL="validation_set"  # primary test set
EVAL_LABEL="eval_set"  # a small eval set
EVAL_LABEL="SemEval2017-task4-test.subtask-A.english.binaryevalformat.balanced"  # OOD test set
EVAL_FILE="${DATA_DIR}/eval_sets/${EVAL_LABEL}.${MODEL_LABEL}.jsonl"


ALPHA=0.95
EXEMPLAR_DIMENSION=1000

MODEL_OUTPUT_DIR=/home/jupyter/models/sdm_paper/release_version/sentiment/"${RUN_SUFFIX_ID}_${MODEL_TYPE}_${ALPHA}_${EXEMPLAR_DIMENSION}"/ # Update with the applicable path

mkdir -p "${MODEL_OUTPUT_DIR}"


LEARNING_RATE=0.00001

# train:

python -u reexpress.py \
--input_training_set_file "${TRAIN_FILE}" \
--input_calibration_set_file "${CALIBRATION_FILE}" \
--input_eval_set_file "${EVAL_FILE}" \
--alpha=${ALPHA} \
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
--main_device="cuda:0" \
--use_embeddings > ${MODEL_OUTPUT_DIR}/run1.log.txt

echo ${MODEL_OUTPUT_DIR}/run1.log.txt


#########################################################################################################
##################### Analysis
#########################################################################################################

cd code/reexpress # Update with the applicable path
conda activate re_mcp_v200


RUN_SUFFIX_ID="mixtral_8x7b"
MODEL_TYPE="classifier"

ALPHA=0.95
EXEMPLAR_DIMENSION=1000
LEARNING_RATE=0.00001

MODEL_OUTPUT_DIR=/home/jupyter/models/sdm_paper/release_version/sentiment/"${RUN_SUFFIX_ID}_${MODEL_TYPE}_${ALPHA}_${EXEMPLAR_DIMENSION}"/ # Update with the applicable path

# run this separately, outside the loop below
EVAL_LABEL=best_iteration_data_calibration
EVAL_FILE="${MODEL_OUTPUT_DIR}/best_iteration_data/calibration.jsonl"
LATEX_DATASET_LABEL="datasetSentiment ca"

DATA_DIR="/home/jupyter/data/classification/sentiment_mixtral_8x7b"  # Update with the applicable path
MODEL_LABEL="mixtral_8x7b"
LATEX_MODEL_NAME='modelMixtralSDM'

#EVAL_LABEL="validation_set"  # primary test set
#EVAL_LABEL="eval_set"  # a small eval set
#EVAL_LABEL="SemEval2017-task4-test.subtask-A.english.binaryevalformat.balanced"  # OOD test set
#EVAL_LABEL="validation_set.ood_random_shuffle"  # primary test set randomly shuffled
#EVAL_LABEL="eval_set.ood_random_shuffle"  # a small eval set randomly shuffled
#EVAL_LABEL="SemEval2017-task4-test.subtask-A.english.binaryevalformat.balanced.ood_random_shuffle"  # OOD test set randomly shuffled
#EVAL_FILE="${DATA_DIR}/eval_sets/${EVAL_LABEL}.${MODEL_LABEL}.jsonl"


MODEL_OUTPUT_DIR_WITH_SUBFOLDER=${MODEL_OUTPUT_DIR}/final_eval_output
mkdir ${MODEL_OUTPUT_DIR_WITH_SUBFOLDER}

for EVAL_LABEL in "validation_set" "eval_set" "SemEval2017-task4-test.subtask-A.english.binaryevalformat.balanced" "validation_set.ood_random_shuffle" "eval_set.ood_random_shuffle" "SemEval2017-task4-test.subtask-A.english.binaryevalformat.balanced.ood_random_shuffle"; do
EVAL_FILE="${DATA_DIR}/eval_sets/${EVAL_LABEL}.${MODEL_LABEL}.jsonl"

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
--alpha=${ALPHA} \
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
--label_error_file=${MODEL_OUTPUT_DIR_WITH_SUBFOLDER}/"eval.${EVAL_LABEL}.possible_label_errors.jsonl" \
--predictions_in_high_reliability_region_file=${MODEL_OUTPUT_DIR_WITH_SUBFOLDER}/"eval.${EVAL_LABEL}.high_reliability.jsonl" \
--prediction_output_file=${MODEL_OUTPUT_DIR_WITH_SUBFOLDER}/"eval.${EVAL_LABEL}.all_predictions.jsonl" \
--eval_only \
--main_device="cuda:0" \
--construct_results_latex_table_rows \
--additional_latex_meta_data="${LATEX_DATASET_LABEL},${LATEX_MODEL_NAME}" > ${MODEL_OUTPUT_DIR_WITH_SUBFOLDER}/"eval.${EVAL_LABEL}.version_2.0.0.log.txt"

echo "Eval Label: ${EVAL_LABEL}"
echo "Possible label errors (sorted) file: "${MODEL_OUTPUT_DIR_WITH_SUBFOLDER}/"eval.${EVAL_LABEL}.possible_label_errors.jsonl"
echo "High reliablity region predictions (sorted) file: "${MODEL_OUTPUT_DIR_WITH_SUBFOLDER}/"eval.${EVAL_LABEL}.high_reliability.jsonl"
echo "All predictions file: "${MODEL_OUTPUT_DIR_WITH_SUBFOLDER}/"eval.${EVAL_LABEL}.all_predictions.jsonl"
echo "Eval log file: "${MODEL_OUTPUT_DIR_WITH_SUBFOLDER}/"eval.${EVAL_LABEL}.version_2.0.0.log.txt"
done



#########################################################################################################
##################### Graph
#########################################################################################################

cd code/reexpress # Update with the applicable path
conda activate re_mcp_v200


RUN_SUFFIX_ID="mixtral_8x7b"
MODEL_TYPE="classifier"

ALPHA=0.95
EXEMPLAR_DIMENSION=1000
LEARNING_RATE=0.00001

MODEL_OUTPUT_DIR=/home/jupyter/models/sdm_paper/release_version/sentiment/"${RUN_SUFFIX_ID}_${MODEL_TYPE}_${ALPHA}_${EXEMPLAR_DIMENSION}"/ # Update with the applicable path

INPUT_DIR="${MODEL_OUTPUT_DIR}/final_eval_output"

# Run each data block in turn

X_BIN_WIDTH=200

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
--model_version_label="v2.0.0" \
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
--model_version_label="v2.0.0" \
--save_file_prefix=${OUTPUT_DIR}/${OUTPUT_FILE_PREFIX}



