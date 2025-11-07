#########################################################################################################
##################### Compute
#########################################################################################################

# 12 GB of GPU memory should be sufficient. (That is a conservative estimate;
# much less is likely needed given a batch size of 50.) This can also be run on CPU by
# setting --main_device="cpu".

#########################################################################################################
##################### Factcheck train and eval; --is_baseline_adaptor
#########################################################################################################


cd code/reexpress # Update with the applicable path
conda activate re_mcp_v200


RUN_SUFFIX_ID="phi_3_5_instruct"
MODEL_TYPE="baseline_adaptor"

DATA_DIR="/home/jupyter/data/classification/factcheck_phi35" # Update with the applicable path

# 'embedding' field is from Phi-3.5 decoder
MODEL_LABEL="phi35"
TRAIN_FILE="${DATA_DIR}/train.${MODEL_LABEL}.jsonl"
CALIBRATION_FILE="${DATA_DIR}/calibration.${MODEL_LABEL}.jsonl"
EVAL_FILE="${DATA_DIR}/ood_eval.${MODEL_LABEL}.jsonl"


ALPHA=0.95
EXEMPLAR_DIMENSION=1000

MODEL_OUTPUT_DIR=/home/jupyter/models/sdm_paper/release_version/factcheck/"${RUN_SUFFIX_ID}_${MODEL_TYPE}_${ALPHA}_${EXEMPLAR_DIMENSION}"/  # Update with the applicable path

mkdir -p "${MODEL_OUTPUT_DIR}"


LEARNING_RATE=0.00001


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

MODEL_OUTPUT_DIR=/home/jupyter/models/sdm_paper/release_version/factcheck/"${RUN_SUFFIX_ID}_${MODEL_TYPE}_${ALPHA}_${EXEMPLAR_DIMENSION}"/  # Update with the applicable path



EVAL_LABEL=best_iteration_data_calibration
EVAL_FILE="${MODEL_OUTPUT_DIR}/best_iteration_data/calibration.jsonl"

# Run each data block in turn
DATA_DIR="/home/jupyter/data/classification/factcheck_phi35" # Update with the applicable path
MODEL_LABEL="phi35"
EVAL_LABEL="ood_eval"  # primary test set (co-variate shifted)
EVAL_FILE="${DATA_DIR}/${EVAL_LABEL}.${MODEL_LABEL}.jsonl"

DATA_DIR="/home/jupyter/data/classification/factcheck_phi35" # Update with the applicable path
MODEL_LABEL="phi35"
EVAL_LABEL="ood_eval.ood_random_shuffle"  # OOD
EVAL_FILE="${DATA_DIR}/${EVAL_LABEL}.${MODEL_LABEL}.jsonl"

MODEL_OUTPUT_DIR_WITH_SUBFOLDER=${MODEL_OUTPUT_DIR}/final_eval_output
mkdir ${MODEL_OUTPUT_DIR_WITH_SUBFOLDER}

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
--is_baseline_adaptor > ${MODEL_OUTPUT_DIR_WITH_SUBFOLDER}/"eval.${EVAL_LABEL}.version_2.0.0.log.txt"

echo "Eval Label: ${EVAL_LABEL}"
echo "All predictions file: "${MODEL_OUTPUT_DIR_WITH_SUBFOLDER}/"eval.${EVAL_LABEL}.all_predictions.jsonl"
echo "Eval log file: "${MODEL_OUTPUT_DIR_WITH_SUBFOLDER}/"eval.${EVAL_LABEL}.version_2.0.0.log.txt"



#########################################################################################################
############################################# conformal and temperature scaling baselines -- factcheck;
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

MODEL_OUTPUT_DIR=/home/jupyter/models/sdm_paper/release_version/factcheck/"${RUN_SUFFIX_ID}_${MODEL_TYPE}_${ALPHA}_${EXEMPLAR_DIMENSION}"/  # Update with the applicable path

INPUT_DIR="${MODEL_OUTPUT_DIR}/final_eval_output"

MODEL_LABEL="phi35"
LATEX_MODEL_NAME="modelPhiThreeFiveInstructCNNAdaptor"
CALIBRATION_FILE=${INPUT_DIR}/eval.best_iteration_data_calibration.all_predictions.jsonl

# Calibration is run multiple times for simplicity, but the calibration thresholds are the same due to the seed.

for EVAL_LABEL in "ood_eval" "ood_eval.ood_random_shuffle"; do

EVAL_FILE=${INPUT_DIR}/"eval.${EVAL_LABEL}.all_predictions.jsonl"

# Set LATEX_DATASET_LABEL based on EVAL_LABEL
if [ "$EVAL_LABEL" = "ood_eval" ]; then
    LATEX_DATASET_LABEL="datasetFactcheck"
elif [ "$EVAL_LABEL" = "ood_eval.ood_random_shuffle" ]; then
    LATEX_DATASET_LABEL="datasetFactcheckShuffled"
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
############################################# conformal and temperature scaling baselines -- factcheck;
### Logits from the linear layer of the underlying LLM
#########################################################################################################

# We include a version of the MIT licensed code of https://github.com/aangelopoulos/conformal-classification at /third_party_baselines/conformal/conformal_classification-master_modified_for_2025_baselines. This does not make any substantive changes to the underlying conformal methods, and exists to simply add a wrapper script `baseline_comp_local.py` to read in the cached output logits from above.


# We also need torchvision and scipy
#conda create --name re_mcp_v200_torchvision --clone re_mcp_v200
#pip install torchvision==0.22.1 --no-deps
#conda install scipy

cd /code/third_party_baselines/conformal/conformal_classification-master_modified_for_2025_baselines  # Update with the applicable path
conda activate re_mcp_v200_torchvision


DATA_DIR="/home/jupyter/data/classification/factcheck_phi35" # Update with the applicable path

# 'embedding' field is from Phi-3.5 decoder
MODEL_LABEL="phi35"
LATEX_MODEL_NAME="modelPhiThreeFiveInstruct"
# LLM does not need a train/calibration split: We combine to form the calibration set, which is what one would do in practice.
#cat "${DATA_DIR}/train.${MODEL_LABEL}.jsonl" "${DATA_DIR}/calibration.${MODEL_LABEL}.jsonl" > "${DATA_DIR}/train_and_calibration.${MODEL_LABEL}.jsonl"

CALIBRATION_FILE="${DATA_DIR}/train_and_calibration.${MODEL_LABEL}.jsonl"

# Calibration is run multiple times for simplicity, but the calibration thresholds are the same due to the seed.

for EVAL_LABEL in "ood_eval" "ood_eval.ood_random_shuffle"; do

EVAL_FILE=${DATA_DIR}/"${EVAL_LABEL}.${MODEL_LABEL}.jsonl"

# Set LATEX_DATASET_LABEL based on EVAL_LABEL
if [ "$EVAL_LABEL" = "ood_eval" ]; then
    LATEX_DATASET_LABEL="datasetFactcheck"
elif [ "$EVAL_LABEL" = "ood_eval.ood_random_shuffle" ]; then
    LATEX_DATASET_LABEL="datasetFactcheckShuffled"
fi

COVERAGE=0.95
L_CRITERION='adaptiveness'
OUTPUT_DIR=/home/jupyter/models/sdm_paper/release_version/factcheck/${MODEL_LABEL}_llm/baseline_calibration_output/
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

