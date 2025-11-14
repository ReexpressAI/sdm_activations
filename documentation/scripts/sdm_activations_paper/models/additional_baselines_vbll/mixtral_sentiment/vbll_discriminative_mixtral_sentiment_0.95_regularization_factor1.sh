#########################################################################################################
##################### Compute
#########################################################################################################

# 12 GB of GPU memory should be sufficient. (That is a conservative estimate;
# much less is likely needed given a batch size of 50.) This can also be run on CPU by
# setting --main_device="cpu".

#########################################################################################################
##################### Install dependencies
#########################################################################################################

#https://github.com/VectorInstitute/vbll

conda create --name re_mcp_v200_vbll_comparison --clone re_mcp_v200

conda activate re_mcp_v200_vbll_comparison

pip install vbll

#Successfully installed vbll-0.4.9

#########################################################################################################
##################### sentiment train and eval; --is_discriminative_vbll_model
#########################################################################################################


#cd code/reexpress # Update with the applicable path
cd /home/jupyter/repos/reexpress_labs_internal/labs/vbll_baseline/code/reexpress

conda activate re_mcp_v200_vbll_comparison


RUN_SUFFIX_ID="mixtral_8x7b"
MODEL_TYPE="vbll_discriminative"

DATA_DIR="/home/jupyter/data/classification/sentiment_mixtral_8x7b" # Update with the applicable path


MODEL_LABEL="mixtral_8x7b"
TRAIN_FILE="${DATA_DIR}/training_set.${MODEL_LABEL}.jsonl"
CALIBRATION_FILE="${DATA_DIR}/calibration_set.${MODEL_LABEL}.jsonl"


EVAL_LABEL="validation_set"  # primary test set
EVAL_LABEL="eval_set"  # a small eval set
EVAL_LABEL="SemEval2017-task4-test.subtask-A.english.binaryevalformat.balanced"  # OOD test set
EVAL_FILE="${DATA_DIR}/eval_sets/${EVAL_LABEL}.${MODEL_LABEL}.jsonl"


ALPHA=0.95
VBLL_DIMENSION=795
VBLL_REGULARIZATION_FACTOR=1.0

MODEL_OUTPUT_DIR=/home/jupyter/models/sdm_paper/release_version_additional/sentiment/"${RUN_SUFFIX_ID}_${MODEL_TYPE}_${ALPHA}_${VBLL_DIMENSION}_reg${VBLL_REGULARIZATION_FACTOR}"/  # Update with the applicable path

mkdir -p "${MODEL_OUTPUT_DIR}"


LEARNING_RATE=0.00001

echo ${MODEL_OUTPUT_DIR}/run1.log.txt

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
--use_embeddings \
--is_discriminative_vbll_model \
--vbll_hidden_dimension=${VBLL_DIMENSION} \
--vbll_regularization_multiplicative_factor=${VBLL_REGULARIZATION_FACTOR} \
--main_device="cuda:1" > ${MODEL_OUTPUT_DIR}/run1.log.txt

#/home/jupyter/models/sdm_paper/release_version_additional/sentiment/mixtral_8x7b_vbll_discriminative_0.95_795_reg1.0//run1.log.txt


#########################################################################################################
##################### Analysis
#########################################################################################################

#cd code/reexpress # Update with the applicable path
cd /home/jupyter/repos/reexpress_labs_internal/labs/vbll_baseline/code/reexpress

conda activate re_mcp_v200_vbll_comparison


RUN_SUFFIX_ID="mixtral_8x7b"
MODEL_TYPE="vbll_discriminative"


ALPHA=0.95
VBLL_DIMENSION=795
VBLL_REGULARIZATION_FACTOR=1.0

MODEL_OUTPUT_DIR=/home/jupyter/models/sdm_paper/release_version_additional/sentiment/"${RUN_SUFFIX_ID}_${MODEL_TYPE}_${ALPHA}_${VBLL_DIMENSION}_reg${VBLL_REGULARIZATION_FACTOR}"/  # Update with the applicable path


LEARNING_RATE=0.00001

MODEL_OUTPUT_DIR_WITH_SUBFOLDER=${MODEL_OUTPUT_DIR}/final_eval_output
mkdir ${MODEL_OUTPUT_DIR_WITH_SUBFOLDER}

DATA_DIR="/home/jupyter/data/classification/sentiment_mixtral_8x7b" # Update with the applicable path
MODEL_LABEL="mixtral_8x7b"
LATEX_MODEL_NAME="modelMixtralDiscVBLLMLP"

for EVAL_LABEL in "best_iteration_data_calibration" "validation_set" "eval_set" "SemEval2017-task4-test.subtask-A.english.binaryevalformat.balanced" "validation_set.ood_random_shuffle" "eval_set.ood_random_shuffle" "SemEval2017-task4-test.subtask-A.english.binaryevalformat.balanced.ood_random_shuffle"; do
EVAL_FILE="${DATA_DIR}/eval_sets/${EVAL_LABEL}.${MODEL_LABEL}.jsonl"

# Set LATEX_DATASET_LABEL based on EVAL_LABEL
if [ "$EVAL_LABEL" = "validation_set" ]; then
    LATEX_DATASET_LABEL="datasetSentiment"
elif [ "$EVAL_LABEL" = "best_iteration_data_calibration" ]; then
    LATEX_DATASET_LABEL="datasetSentiment calibrationSplit"
    EVAL_FILE="${MODEL_OUTPUT_DIR}/best_iteration_data/calibration.jsonl"
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
--prediction_output_file=${MODEL_OUTPUT_DIR_WITH_SUBFOLDER}/"eval.${EVAL_LABEL}.all_predictions.jsonl" \
--eval_only \
--is_discriminative_vbll_model \
--vbll_hidden_dimension=${VBLL_DIMENSION} \
--vbll_regularization_multiplicative_factor=${VBLL_REGULARIZATION_FACTOR} \
--main_device="cuda:1" \
--construct_results_latex_table_rows \
--additional_latex_meta_data="${LATEX_DATASET_LABEL},${LATEX_MODEL_NAME}" > ${MODEL_OUTPUT_DIR_WITH_SUBFOLDER}/"eval.${EVAL_LABEL}.version_2.0.0.log.txt"

echo "Eval Label: ${EVAL_LABEL}"
echo "All predictions file: "${MODEL_OUTPUT_DIR_WITH_SUBFOLDER}/"eval.${EVAL_LABEL}.all_predictions.jsonl"
echo "Eval log file: "${MODEL_OUTPUT_DIR_WITH_SUBFOLDER}/"eval.${EVAL_LABEL}.version_2.0.0.log.txt"

done


#Eval Label: best_iteration_data_calibration
#All predictions file: /home/jupyter/models/sdm_paper/release_version_additional/sentiment/mixtral_8x7b_vbll_discriminative_0.95_795_reg1.0//final_eval_output/eval.best_iteration_data_calibration.all_predictions.jsonl
#Eval log file: /home/jupyter/models/sdm_paper/release_version_additional/sentiment/mixtral_8x7b_vbll_discriminative_0.95_795_reg1.0//final_eval_output/eval.best_iteration_data_calibration.version_2.0.0.log.txt
#
#Eval Label: validation_set
#All predictions file: /home/jupyter/models/sdm_paper/release_version_additional/sentiment/mixtral_8x7b_vbll_discriminative_0.95_795_reg1.0//final_eval_output/eval.validation_set.all_predictions.jsonl
#Eval log file: /home/jupyter/models/sdm_paper/release_version_additional/sentiment/mixtral_8x7b_vbll_discriminative_0.95_795_reg1.0//final_eval_output/eval.validation_set.version_2.0.0.log.txt
#
#Eval Label: eval_set
#All predictions file: /home/jupyter/models/sdm_paper/release_version_additional/sentiment/mixtral_8x7b_vbll_discriminative_0.95_795_reg1.0//final_eval_output/eval.eval_set.all_predictions.jsonl
#Eval log file: /home/jupyter/models/sdm_paper/release_version_additional/sentiment/mixtral_8x7b_vbll_discriminative_0.95_795_reg1.0//final_eval_output/eval.eval_set.version_2.0.0.log.txt
#
#Eval Label: SemEval2017-task4-test.subtask-A.english.binaryevalformat.balanced
#All predictions file: /home/jupyter/models/sdm_paper/release_version_additional/sentiment/mixtral_8x7b_vbll_discriminative_0.95_795_reg1.0//final_eval_output/eval.SemEval2017-task4-test.subtask-A.english.binaryevalformat.balanced.all_predictions.jsonl
#Eval log file: /home/jupyter/models/sdm_paper/release_version_additional/sentiment/mixtral_8x7b_vbll_discriminative_0.95_795_reg1.0//final_eval_output/eval.SemEval2017-task4-test.subtask-A.english.binaryevalformat.balanced.version_2.0.0.log.txt
#
#Eval Label: validation_set.ood_random_shuffle
#All predictions file: /home/jupyter/models/sdm_paper/release_version_additional/sentiment/mixtral_8x7b_vbll_discriminative_0.95_795_reg1.0//final_eval_output/eval.validation_set.ood_random_shuffle.all_predictions.jsonl
#Eval log file: /home/jupyter/models/sdm_paper/release_version_additional/sentiment/mixtral_8x7b_vbll_discriminative_0.95_795_reg1.0//final_eval_output/eval.validation_set.ood_random_shuffle.version_2.0.0.log.txt
#
#Eval Label: eval_set.ood_random_shuffle
#All predictions file: /home/jupyter/models/sdm_paper/release_version_additional/sentiment/mixtral_8x7b_vbll_discriminative_0.95_795_reg1.0//final_eval_output/eval.eval_set.ood_random_shuffle.all_predictions.jsonl
#Eval log file: /home/jupyter/models/sdm_paper/release_version_additional/sentiment/mixtral_8x7b_vbll_discriminative_0.95_795_reg1.0//final_eval_output/eval.eval_set.ood_random_shuffle.version_2.0.0.log.txt
#
#Eval Label: SemEval2017-task4-test.subtask-A.english.binaryevalformat.balanced.ood_random_shuffle
#All predictions file: /home/jupyter/models/sdm_paper/release_version_additional/sentiment/mixtral_8x7b_vbll_discriminative_0.95_795_reg1.0//final_eval_output/eval.SemEval2017-task4-test.subtask-A.english.binaryevalformat.balanced.ood_random_shuffle.all_predictions.jsonl
#Eval log file: /home/jupyter/models/sdm_paper/release_version_additional/sentiment/mixtral_8x7b_vbll_discriminative_0.95_795_reg1.0//final_eval_output/eval.SemEval2017-task4-test.subtask-A.english.binaryevalformat.balanced.ood_random_shuffle.version_2.0.0.log.txt


