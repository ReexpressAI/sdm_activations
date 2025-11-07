#########################################################################################################
##################### Compute
#########################################################################################################

# 12 GB of GPU memory should be sufficient. (That is a conservative estimate;
# much less is likely needed given a batch size of 50.) This can also be run on CPU by
# setting --main_device="cpu".

#########################################################################################################
##################### Sentiment train and eval; --is_baseline_adaptor
#########################################################################################################


cd code/reexpress # Update with the applicable path
conda activate re_mcp_v200


RUN_SUFFIX_ID="phi_3_5_instruct"
MODEL_TYPE="baseline_adaptor"

DATA_DIR="/home/jupyter/data/classification/sentiment_phi35"  # Update with the applicable path

# 'embedding' field is from Phi-3.5 decoder
MODEL_LABEL="phi35"
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
--use_embeddings \
--is_baseline_adaptor \
--main_device="cuda:0" > ${MODEL_OUTPUT_DIR}/run1.log.txt

echo ${MODEL_OUTPUT_DIR}/run1.log.txt


#########################################################################################################
##################### Analysis
#########################################################################################################

cd code/reexpress # Update with the applicable path
conda activate re_mcp_v200


RUN_SUFFIX_ID="phi_3_5_instruct"
MODEL_TYPE="baseline_adaptor"

ALPHA=0.95
EXEMPLAR_DIMENSION=1000
LEARNING_RATE=0.00001

MODEL_OUTPUT_DIR=/home/jupyter/models/sdm_paper/release_version/sentiment/"${RUN_SUFFIX_ID}_${MODEL_TYPE}_${ALPHA}_${EXEMPLAR_DIMENSION}"/ # Update with the applicable path

# run this separately, outside the loop below
EVAL_LABEL=best_iteration_data_calibration
EVAL_FILE="${MODEL_OUTPUT_DIR}/best_iteration_data/calibration.jsonl"

DATA_DIR="/home/jupyter/data/classification/sentiment_phi35"  # Update with the applicable path
MODEL_LABEL="phi35"

EVAL_LABEL="validation_set"  # primary test set
EVAL_LABEL="eval_set"  # a small eval set
EVAL_LABEL="SemEval2017-task4-test.subtask-A.english.binaryevalformat.balanced"  # OOD test set
EVAL_LABEL="validation_set.ood_random_shuffle"  # primary test set randomly shuffled
EVAL_LABEL="eval_set.ood_random_shuffle"  # a small eval set randomly shuffled
EVAL_LABEL="SemEval2017-task4-test.subtask-A.english.binaryevalformat.balanced.ood_random_shuffle"  # OOD test set randomly shuffled
EVAL_FILE="${DATA_DIR}/eval_sets/${EVAL_LABEL}.${MODEL_LABEL}.jsonl"


MODEL_OUTPUT_DIR_WITH_SUBFOLDER=${MODEL_OUTPUT_DIR}/final_eval_output
mkdir ${MODEL_OUTPUT_DIR_WITH_SUBFOLDER}

for EVAL_LABEL in "validation_set" "eval_set" "SemEval2017-task4-test.subtask-A.english.binaryevalformat.balanced" "validation_set.ood_random_shuffle" "eval_set.ood_random_shuffle" "SemEval2017-task4-test.subtask-A.english.binaryevalformat.balanced.ood_random_shuffle"; do
EVAL_FILE="${DATA_DIR}/eval_sets/${EVAL_LABEL}.${MODEL_LABEL}.jsonl"

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
--prediction_output_file=${MODEL_OUTPUT_DIR_WITH_SUBFOLDER}/"eval.${EVAL_LABEL}.all_predictions.jsonl" \
--eval_only \
--main_device="cuda:0" \
--is_baseline_adaptor > ${MODEL_OUTPUT_DIR_WITH_SUBFOLDER}/"eval.${EVAL_LABEL}.version_2.0.0.log.txt"

echo "Eval Label: ${EVAL_LABEL}"
echo "All predictions file: "${MODEL_OUTPUT_DIR_WITH_SUBFOLDER}/"eval.${EVAL_LABEL}.all_predictions.jsonl"
echo "Eval log file: "${MODEL_OUTPUT_DIR_WITH_SUBFOLDER}/"eval.${EVAL_LABEL}.version_2.0.0.log.txt"
done



#########################################################################################################
############################################# conformal and temperature scaling baselines -- sentiment;
### From adaptor layer
#########################################################################################################

# We include a version of the MIT licensed code of https://github.com/aangelopoulos/conformal-classification at /third_party_baselines/conformal/conformal_classification-master_modified_for_2025_baselines. This does not make any substantive changes to the underlying conformal methods, and exists to simply add a wrapper script `baseline_comp_local.py` to read in the cached output logits from above.


# We also need torchvision and scipy
#conda create --name re_mcp_v200_torchvision --clone re_mcp_v200
#pip install torchvision==0.22.1 --no-deps
#conda install scipy

cd /code/third_party_baselines/conformal/conformal_classification-master_modified_for_2025_baselines  # Update with the applicable path
conda activate re_mcp_v200_torchvision


RUN_SUFFIX_ID="phi_3_5_instruct"
MODEL_TYPE="baseline_adaptor"

ALPHA=0.95
EXEMPLAR_DIMENSION=1000
LEARNING_RATE=0.00001

MODEL_OUTPUT_DIR=/home/jupyter/models/sdm_paper/release_version/sentiment/"${RUN_SUFFIX_ID}_${MODEL_TYPE}_${ALPHA}_${EXEMPLAR_DIMENSION}"/ # Update with the applicable path

INPUT_DIR="${MODEL_OUTPUT_DIR}/final_eval_output"

MODEL_LABEL="phi35"
LATEX_MODEL_NAME="modelPhiThreeFiveInstructCNNAdaptor"

CALIBRATION_FILE=${INPUT_DIR}/eval.best_iteration_data_calibration.all_predictions.jsonl

# Calibration is run multiple times for simplicity, but the calibration thresholds are the same due to the seed.


for EVAL_LABEL in "validation_set" "eval_set" "SemEval2017-task4-test.subtask-A.english.binaryevalformat.balanced" "validation_set.ood_random_shuffle" "eval_set.ood_random_shuffle" "SemEval2017-task4-test.subtask-A.english.binaryevalformat.balanced.ood_random_shuffle"; do

EVAL_FILE=${INPUT_DIR}/"eval.${EVAL_LABEL}.all_predictions.jsonl"

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


COVERAGE=0.95
L_CRITERION='adaptiveness'
OUTPUT_DIR=${MODEL_OUTPUT_DIR}/baseline_calibration_output/
mkdir ${OUTPUT_DIR}

python -u baseline_comp_local.py \
--calibration_files ${CALIBRATION_FILE} \
--eval_files ${EVAL_FILE} \
--batch_size 50 \
--seed 0 \
--number_of_classes 2 \
--empirical_coverage ${COVERAGE} \
--lambda_criterion ${L_CRITERION} \
--probability_threshold ${COVERAGE} \
--additional_latex_meta_data="${LATEX_DATASET_LABEL},${LATEX_MODEL_NAME}" > ${OUTPUT_DIR}/${EVAL_LABEL}.${MODEL_LABEL}.${COVERAGE}_${L_CRITERION}.raps_baseline.log.txt
echo ${OUTPUT_DIR}/${EVAL_LABEL}.${MODEL_LABEL}.${COVERAGE}_${L_CRITERION}.raps_baseline.log.txt

python -u baseline_comp_local.py \
--calibration_files ${CALIBRATION_FILE} \
--eval_files ${EVAL_FILE} \
--batch_size 50 \
--seed 0 \
--number_of_classes 2 \
--empirical_coverage ${COVERAGE} \
--lambda_criterion ${L_CRITERION} \
--run_aps_baseline \
--probability_threshold ${COVERAGE} \
--additional_latex_meta_data="${LATEX_DATASET_LABEL},${LATEX_MODEL_NAME}" > ${OUTPUT_DIR}/${EVAL_LABEL}.${MODEL_LABEL}.${COVERAGE}_aps_baseline.log.txt
echo ${OUTPUT_DIR}/${EVAL_LABEL}.${MODEL_LABEL}.${COVERAGE}_aps_baseline.log.txt

done



#########################################################################################################
############################################# conformal and temperature scaling baselines -- sentiment;
### Logits from the linear layer of the underlying LLM
#########################################################################################################

# We include a version of the MIT licensed code of https://github.com/aangelopoulos/conformal-classification at /third_party_baselines/conformal/conformal_classification-master_modified_for_2025_baselines. This does not make any substantive changes to the underlying conformal methods, and exists to simply add a wrapper script `baseline_comp_local.py` to read in the cached output logits from above.

# We also need torchvision and scipy
#conda create --name re_mcp_v200_torchvision --clone re_mcp_v200
#pip install torchvision==0.22.1 --no-deps
#conda install scipy

cd /code/third_party_baselines/conformal/conformal_classification-master_modified_for_2025_baselines  # Update with the applicable path
conda activate re_mcp_v200_torchvision


DATA_DIR="/home/jupyter/data/classification/sentiment_phi35" # Update with the applicable path

# 'embedding' field is from Phi-3.5 decoder
MODEL_LABEL="phi35"
LATEX_MODEL_NAME="modelPhiThreeFiveInstruct"
# LLM does not need a train/calibration split: We combine to form the calibration set, which is what one would do in practice.
#cat "${DATA_DIR}/training_set.${MODEL_LABEL}.jsonl" "${DATA_DIR}/calibration_set.${MODEL_LABEL}.jsonl" > "${DATA_DIR}/train_and_calibration.${MODEL_LABEL}.jsonl"

CALIBRATION_FILE="${DATA_DIR}/train_and_calibration.${MODEL_LABEL}.jsonl"

# Calibration is run multiple times for simplicity, but the calibration thresholds are the same due to the seed.

for EVAL_LABEL in "validation_set" "eval_set" "SemEval2017-task4-test.subtask-A.english.binaryevalformat.balanced" "validation_set.ood_random_shuffle" "eval_set.ood_random_shuffle" "SemEval2017-task4-test.subtask-A.english.binaryevalformat.balanced.ood_random_shuffle"; do

EVAL_FILE=${DATA_DIR}/eval_sets/"${EVAL_LABEL}.${MODEL_LABEL}.jsonl"

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

COVERAGE=0.95
L_CRITERION='adaptiveness'
OUTPUT_DIR=/home/jupyter/models/sdm_paper/release_version/sentiment/${MODEL_LABEL}_llm/baseline_calibration_output/
mkdir -p ${OUTPUT_DIR}

python -u baseline_comp_local.py \
--prediction_field_name="llm_classification" \
--calibration_files ${CALIBRATION_FILE} \
--eval_files ${EVAL_FILE} \
--batch_size 50 \
--seed 0 \
--number_of_classes 2 \
--empirical_coverage ${COVERAGE} \
--lambda_criterion ${L_CRITERION} \
--probability_threshold ${COVERAGE} \
--additional_latex_meta_data="${LATEX_DATASET_LABEL},${LATEX_MODEL_NAME}" > ${OUTPUT_DIR}/${EVAL_LABEL}.${MODEL_LABEL}.${COVERAGE}_${L_CRITERION}.raps_baseline.log.txt
echo ${OUTPUT_DIR}/${EVAL_LABEL}.${MODEL_LABEL}.${COVERAGE}_${L_CRITERION}.raps_baseline.log.txt

python -u baseline_comp_local.py \
--prediction_field_name="llm_classification" \
--calibration_files ${CALIBRATION_FILE} \
--eval_files ${EVAL_FILE} \
--batch_size 50 \
--seed 0 \
--number_of_classes 2 \
--empirical_coverage ${COVERAGE} \
--lambda_criterion ${L_CRITERION} \
--run_aps_baseline \
--probability_threshold ${COVERAGE} \
--additional_latex_meta_data="${LATEX_DATASET_LABEL},${LATEX_MODEL_NAME}" > ${OUTPUT_DIR}/${EVAL_LABEL}.${MODEL_LABEL}.${COVERAGE}_aps_baseline.log.txt
echo ${OUTPUT_DIR}/${EVAL_LABEL}.${MODEL_LABEL}.${COVERAGE}_aps_baseline.log.txt

done

