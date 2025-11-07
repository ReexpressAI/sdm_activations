#########################################################################################################
## GPUs used for this experiment
#########################################################################################################

#+-----------------------------------------------------------------------------------------+
#| NVIDIA-SMI 550.90.07              Driver Version: 550.90.07      CUDA Version: 12.4     |
#|-----------------------------------------+------------------------+----------------------+
#| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |
#| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |
#|                                         |                        |               MIG M. |
#|=========================================+========================+======================|
#|   0  NVIDIA A100-SXM4-80GB          On  |   00000000:00:05.0 Off |                    0 |
#| N/A   30C    P0             65W /  400W |       1MiB /  81920MiB |      0%      Default |
#|                                         |                        |             Disabled |
#+-----------------------------------------+------------------------+----------------------+
#|   1  NVIDIA A100-SXM4-80GB          On  |   00000000:00:06.0 Off |                    0 |
#| N/A   32C    P0             65W /  400W |       1MiB /  81920MiB |      0%      Default |
#|                                         |                        |             Disabled |
#+-----------------------------------------+------------------------+----------------------+
#
#+-----------------------------------------------------------------------------------------+
#| Processes:                                                                              |
#|  GPU   GI   CI        PID   Type   Process name                              GPU Memory |
#|        ID   ID                                                               Usage      |
#|=========================================================================================|
#|  No running processes found                                                             |
#+-----------------------------------------------------------------------------------------+

#########################################################################################################
##################### In this case, we use the base model without fine-tuning.
#########################################################################################################

python sdm_network_simple_chat.py "microsoft/Phi-3.5-mini-instruct" 0


#########################################################################################################
##################### Decoding of the calibration set in preparation for training the final verification layer.
##################### In this case, no verification layers yet exist, so we use the
##################### argument --no_verification_model
##################### Hard negatives derived from the calibration set are then used to train the final verification layer.
#########################################################################################################

cd code/reexpress  # Update with the applicable path

conda activate basetrain
export HF_HOME=/home/jupyter/models/hf

MODEL_LABEL_NAME="phi35_sdm_verification_layer"

RUN_ID="word_ordering_baseline_no_finetuning_suffix_size3_run1"
MODEL_DIR="/home/jupyter/models/order/preview__${MODEL_LABEL_NAME}_${RUN_ID}/models"
OUTPUT_DIR="/home/jupyter/models/order/preview__${MODEL_LABEL_NAME}_${RUN_ID}/outputs"

# Create directories
mkdir -p $DATA_DIR $MODEL_DIR $OUTPUT_DIR

SUFFIX_SIZE=3

# Decoding of the calibration set for use in training the final verification layer:
DATA_SET="wikipedia"
DATA_DIR="/home/jupyter/data/order/${DATA_SET}_sentences_processed_suffix_size${SUFFIX_SIZE}/"
FILE_NAME="calibration"
CHECKPOINT=""
ADDITIONAL_ID=".${DATA_SET}"
SEARCH_DEPTH=0
TEMPERATURE=0.0
LINE_LIMIT="-1"
RUN_LABEL="r1.suffix_size${SUFFIX_SIZE}.searchdepth${SEARCH_DEPTH}.temperature${TEMPERATURE}${ADDITIONAL_ID}.line_limit${LINE_LIMIT}"

# Verification layer parameters
ALPHA=0.95
BEST_EVAL_STEP="baseline_model"  # just a placeholder here

OUTPUT_LOG_FILE=$OUTPUT_DIR/decode.${FILE_NAME}.${RUN_LABEL}.${BEST_EVAL_STEP}.log.txt
echo ${OUTPUT_LOG_FILE}


# Note the use of --no_verification_model and absence of --save_embedding, since no test-time sdm layer has yet been initialized.

accelerate launch --num_processes=2 sdm_network_inference_with_verification.py \
    --line_limit=${LINE_LIMIT} \
    --model_path="microsoft/Phi-3.5-mini-instruct" \
    --no_verification_model \
    --input_file $DATA_DIR/${FILE_NAME}.jsonl \
    --output_file $OUTPUT_DIR/predictions.${FILE_NAME}.${RUN_LABEL}.${BEST_EVAL_STEP}.jsonl \
    --bf16 \
    --search_depth=${SEARCH_DEPTH} \
    --temperature=${TEMPERATURE} \
    --use_file_prompt > ${OUTPUT_LOG_FILE}
    

EVAL_FILE=$OUTPUT_DIR/predictions.${FILE_NAME}.${RUN_LABEL}.${BEST_EVAL_STEP}.jsonl
OUTPUT_EVAL_LOG_FILE=$OUTPUT_DIR/surface_level_eval.${FILE_NAME}.${RUN_LABEL}.${BEST_EVAL_STEP}.log.txt
echo ${OUTPUT_EVAL_LOG_FILE}


python sdm_network_evaluate.py \
    --predictions_file=${EVAL_FILE} \
    --output_file $OUTPUT_DIR/sdm.metrics.${FILE_NAME}.${RUN_LABEL}.jsonl \
    --errors_file $OUTPUT_DIR/sdm.errors.${FILE_NAME}.${RUN_LABEL}.jsonl \
    --verbatim_errors_file $OUTPUT_DIR/sdm.verbatim_errors.${FILE_NAME}.${RUN_LABEL}.jsonl \
    --unparseable_errors_file $OUTPUT_DIR/sdm.unparseable_errors.${FILE_NAME}.${RUN_LABEL}.jsonl \
    --verbose \
    --evaluate_first_sentences > ${OUTPUT_EVAL_LOG_FILE}

#Total examples: 5000
#Verbatim exact matches (including formatting): 883 (17.66%)
#Exact matches (content only, first available tagged sentence): 943 (18.86%)
#No response: 1387 (27.74%)
#
#Additional reference counts among parseable:
#        verification_tp: 943 (18.86%)
#        verification_fp: 2580 (51.60%)
#        verification_fn: 0 (0.00%)
#        verification_tn: 90 (1.80%)
        
# convert the file to the hard negatives format:

HARD_NEGATIVES_FILE=${EVAL_FILE}.processed_hard_negatives.jsonl

python -u sdm_network_process_generation_output_into_hard_negative_format.py \
--input_file=${EVAL_FILE} \
--output_hard_negatives_jsonl_file=${HARD_NEGATIVES_FILE}

#Documents with generated hard negatives: 2807


# Train the final verification layer


CHECKPOINT=""
# Verification layer parameters
ALPHA=0.95
EXEMPLAR_DIMENSION=1000
SDM_LEARNING_RATE=0.00001
SDM_EPOCHS=200

echo ${OUTPUT_DIR}/verification_layer_final_training.run1.log.txt


# in this case, we also need to make the verification directory, since it has not been initialized during a fine-tuning run
mkdir $MODEL_DIR/${MODEL_LABEL_NAME}/verification_model

## Train the verification layer from the best checkpoint.
# Note the use of --sdm_input_calibration_set_file instead of --sdm_input_training_set_file $DATA_DIR/train.jsonl \
# and --use_baseline_model

accelerate launch --num_processes=2 sdm_network_train_verification_layer.py \
    --use_baseline_model \
    --ignore_default_negative_if_generated_hard_negative_is_available \
    --hard_negatives_file=${HARD_NEGATIVES_FILE} \
    --sdm_input_calibration_set_file $DATA_DIR/${FILE_NAME}.jsonl \
    --output_dir $MODEL_DIR/${MODEL_LABEL_NAME}/${CHECKPOINT} \
    --verification_model_dir=$MODEL_DIR/${MODEL_LABEL_NAME}/verification_model \
    --bf16 \
    --mask_prefix \
    --seed 0 \
    --sdm_alpha=${ALPHA} \
    --sdm_class_size 2 \
    --sdm_epoch=${SDM_EPOCHS} \
    --sdm_batch_size 50 \
    --sdm_eval_batch_size 500 \
    --sdm_learning_rate ${SDM_LEARNING_RATE} \
    --sdm_number_of_random_shuffles 1 \
    --sdm_maxQAvailableFromIndexer 2048 \
    --sdm_exemplar_vector_dimension ${EXEMPLAR_DIMENSION} > ${OUTPUT_DIR}/verification_layer_final_training.run1.log.txt

 
    
#########################################################################################################
##################### Inference
#########################################################################################################

cd code/reexpress  # Update with the applicable path

conda activate basetrain
export HF_HOME=/home/jupyter/models/hf

MODEL_LABEL_NAME="phi35_sdm_verification_layer"

RUN_ID="word_ordering_baseline_no_finetuning_suffix_size3_run1"
MODEL_DIR="/home/jupyter/models/order/preview__${MODEL_LABEL_NAME}_${RUN_ID}/models"
OUTPUT_DIR="/home/jupyter/models/order/preview__${MODEL_LABEL_NAME}_${RUN_ID}/outputs"


# Run each dataset block in turn:
SUFFIX_SIZE=3
DATA_SET="harvard"
DATA_DIR="/home/jupyter/data/order/${DATA_SET}_sentences_processed_suffix_size${SUFFIX_SIZE}/"
FILE_NAME="combined"
CHECKPOINT=""
ADDITIONAL_ID=".${DATA_SET}"
SEARCH_DEPTH=0
TEMPERATURE=0.0
LINE_LIMIT="-1"
RUN_LABEL="r1.suffix_size${SUFFIX_SIZE}.searchdepth${SEARCH_DEPTH}.temperature${TEMPERATURE}${ADDITIONAL_ID}.line_limit${LINE_LIMIT}"


#DATA_SET="wikipedia"
#SUFFIX_SIZE=3
#DATA_DIR="/home/jupyter/data/order/${DATA_SET}_sentences_processed_suffix_size${SUFFIX_SIZE}/"
#FILE_NAME="test"
#CHECKPOINT=""
#ADDITIONAL_ID=".${DATA_SET}"
#SEARCH_DEPTH=0
#TEMPERATURE=0.0
#LINE_LIMIT="-1"
#RUN_LABEL="r1.suffix_size${SUFFIX_SIZE}.searchdepth${SEARCH_DEPTH}.temperature${TEMPERATURE}${ADDITIONAL_ID}.line_limit${LINE_LIMIT}"

#DATA_SET="wikipedia"
#SUFFIX_SIZE=3
#DATA_DIR="/home/jupyter/data/order/${DATA_SET}_sentences_processed_suffix_size${SUFFIX_SIZE}/"
#FILE_NAME="challenge_test"
#CHECKPOINT=""
#ADDITIONAL_ID=".${DATA_SET}"
#SEARCH_DEPTH=0
#TEMPERATURE=0.0
#LINE_LIMIT="-1"
#RUN_LABEL="r1.suffix_size${SUFFIX_SIZE}.searchdepth${SEARCH_DEPTH}.temperature${TEMPERATURE}${ADDITIONAL_ID}.line_limit${LINE_LIMIT}"

# Verification layer parameters
ALPHA=0.95

echo $OUTPUT_DIR/decode.${FILE_NAME}.${RUN_LABEL}.log.txt


accelerate launch --num_processes=2 sdm_network_inference_with_verification.py \
    --line_limit=${LINE_LIMIT} \
    --model_path="microsoft/Phi-3.5-mini-instruct" \
    --verification_model_dir=$MODEL_DIR/${MODEL_LABEL_NAME}/verification_model/final \
    --input_file $DATA_DIR/${FILE_NAME}.jsonl \
    --output_file $OUTPUT_DIR/predictions.${FILE_NAME}.${RUN_LABEL}.jsonl \
    --save_embedding \
    --bf16 \
    --search_depth=${SEARCH_DEPTH} \
    --temperature=${TEMPERATURE} \
    --use_file_prompt > $OUTPUT_DIR/decode.${FILE_NAME}.${RUN_LABEL}.log.txt

EVAL_FILE=$OUTPUT_DIR/predictions.${FILE_NAME}.${RUN_LABEL}.jsonl
echo $OUTPUT_DIR/surface_level_eval.${FILE_NAME}.${RUN_LABEL}.log.txt


python sdm_network_evaluate.py \
    --predictions_file=${EVAL_FILE} \
    --output_file $OUTPUT_DIR/sdm.metrics.${FILE_NAME}.${RUN_LABEL}.jsonl \
    --errors_file $OUTPUT_DIR/sdm.errors.${FILE_NAME}.${RUN_LABEL}.jsonl \
    --verbatim_errors_file $OUTPUT_DIR/sdm.verbatim_errors.${FILE_NAME}.${RUN_LABEL}.jsonl \
    --unparseable_errors_file $OUTPUT_DIR/sdm.unparseable_errors.${FILE_NAME}.${RUN_LABEL}.jsonl \
    --verbose \
    --evaluate_first_sentences > $OUTPUT_DIR/surface_level_eval.${FILE_NAME}.${RUN_LABEL}.log.txt

#/home/jupyter/models/order/preview__phi35_sdm_verification_layer_word_ordering_baseline_no_finetuning_suffix_size3_run1/outputs/surface_level_eval.combined.r1.suffix_size3.searchdepth0.temperature0.0.harvard.line_limit-1.log.txt

    #Total examples: 720
    #Verbatim exact matches (including formatting): 237 (32.92%)
    #Exact matches (content only, first available tagged sentence): 297 (41.25%)
    #No response: 21 (2.92%)
    #
    #Additional reference counts among parseable:
    #        verification_tp: 297 (41.25%)
    #        verification_fp: 361 (50.14%)
    #        verification_fn: 0 (0.00%)
    #        verification_tn: 41 (5.69%)
    #
    #Word-level accuracy: 0.6319 ± 0.4135
    #Average attempts: 2.43 (max: 11)

#/home/jupyter/models/order/preview__phi35_sdm_verification_layer_word_ordering_baseline_no_finetuning_suffix_size3_run1/outputs/surface_level_eval.test.r1.suffix_size3.searchdepth0.temperature0.0.wikipedia.line_limit-1.log.txt

    #Total examples: 2000
    #Verbatim exact matches (including formatting): 492 (24.60%)
    #Exact matches (content only, first available tagged sentence): 525 (26.25%)
    #No response: 92 (4.60%)
    #
    #Additional reference counts among parseable:
    #        verification_tp: 525 (26.25%)
    #        verification_fp: 1335 (66.75%)
    #        verification_fn: 0 (0.00%)
    #        verification_tn: 48 (2.40%)
    #
    #Word-level accuracy: 0.5155 ± 0.4586
    #Average attempts: 1.88 (max: 19)

#/home/jupyter/models/order/preview__phi35_sdm_verification_layer_word_ordering_baseline_no_finetuning_suffix_size3_run1/outputs/surface_level_eval.challenge_test.r1.suffix_size3.searchdepth0.temperature0.0.wikipedia.line_limit-1.log.txt

    #Total examples: 2000
    #Verbatim exact matches (including formatting): 250 (12.50%)
    #Exact matches (content only, first available tagged sentence): 285 (14.25%)
    #No response: 102 (5.10%)
    #
    #Additional reference counts among parseable:
    #        verification_tp: 285 (14.25%)
    #        verification_fp: 1593 (79.65%)
    #        verification_fn: 0 (0.00%)
    #        verification_tn: 20 (1.00%)
    #
    #Word-level accuracy: 0.3329 ± 0.4609
    #Average attempts: 1.65 (max: 13)

#########################################################################################################
##################### Eval of binary classification of instruction-following using the test-time SDM layer
#########################################################################################################

cd code/reexpress  # Update with the applicable path

conda activate basetrain
export HF_HOME=/home/jupyter/models/hf

MODEL_LABEL_NAME="phi35_sdm_verification_layer"

RUN_ID="word_ordering_baseline_no_finetuning_suffix_size3_run1"
MODEL_DIR="/home/jupyter/models/order/preview__${MODEL_LABEL_NAME}_${RUN_ID}/models"
OUTPUT_DIR="/home/jupyter/models/order/preview__${MODEL_LABEL_NAME}_${RUN_ID}/outputs"


# Run each dataset block in turn:
SUFFIX_SIZE=3
DATA_SET="harvard"
DATA_DIR="/home/jupyter/data/order/${DATA_SET}_sentences_processed_suffix_size${SUFFIX_SIZE}/"
FILE_NAME="combined"
CHECKPOINT=""
ADDITIONAL_ID=".${DATA_SET}"
SEARCH_DEPTH=0
TEMPERATURE=0.0
LINE_LIMIT="-1"
RUN_LABEL="r1.suffix_size${SUFFIX_SIZE}.searchdepth${SEARCH_DEPTH}.temperature${TEMPERATURE}${ADDITIONAL_ID}.line_limit${LINE_LIMIT}"


#DATA_SET="wikipedia"
#SUFFIX_SIZE=3
#DATA_DIR="/home/jupyter/data/order/${DATA_SET}_sentences_processed_suffix_size${SUFFIX_SIZE}/"
#FILE_NAME="test"
#CHECKPOINT=""
#ADDITIONAL_ID=".${DATA_SET}"
#SEARCH_DEPTH=0
#TEMPERATURE=0.0
#LINE_LIMIT="-1"
#RUN_LABEL="r1.suffix_size${SUFFIX_SIZE}.searchdepth${SEARCH_DEPTH}.temperature${TEMPERATURE}${ADDITIONAL_ID}.line_limit${LINE_LIMIT}"

#DATA_SET="wikipedia"
#SUFFIX_SIZE=3
#DATA_DIR="/home/jupyter/data/order/${DATA_SET}_sentences_processed_suffix_size${SUFFIX_SIZE}/"
#FILE_NAME="challenge_test"
#CHECKPOINT=""
#ADDITIONAL_ID=".${DATA_SET}"
#SEARCH_DEPTH=0
#TEMPERATURE=0.0
#LINE_LIMIT="-1"
#RUN_LABEL="r1.suffix_size${SUFFIX_SIZE}.searchdepth${SEARCH_DEPTH}.temperature${TEMPERATURE}${ADDITIONAL_ID}.line_limit${LINE_LIMIT}"


# Verification layer parameters
ALPHA=0.95

MODEL_OUTPUT_DIR_WITH_SUBFOLDER=${OUTPUT_DIR}/final_eval_output
mkdir ${MODEL_OUTPUT_DIR_WITH_SUBFOLDER}

LATEX_MODEL_NAME="sdmNet"
EVAL_LABEL="${FILE_NAME}.${RUN_LABEL}"

python -u reexpress.py \
--input_eval_set_file=$OUTPUT_DIR/predictions.${FILE_NAME}.${RUN_LABEL}.jsonl \
--use_embeddings \
--alpha=${ALPHA} \
--class_size 2 \
--seed_value 0 \
--eval_batch_size 500 \
--model_dir=$MODEL_DIR/${MODEL_LABEL_NAME}/verification_model/final \
--main_device="cuda:0" \
--label_error_file=${MODEL_OUTPUT_DIR_WITH_SUBFOLDER}/"eval.${EVAL_LABEL}.possible_label_errors.jsonl" \
--predictions_in_high_reliability_region_file=${MODEL_OUTPUT_DIR_WITH_SUBFOLDER}/"eval.${EVAL_LABEL}.high_reliability.jsonl" \
--prediction_output_file=${MODEL_OUTPUT_DIR_WITH_SUBFOLDER}/"eval.${EVAL_LABEL}.all_predictions.jsonl" \
--eval_only \
--construct_results_latex_table_rows \
--additional_latex_meta_data="${EVAL_LABEL},${LATEX_MODEL_NAME}"> ${MODEL_OUTPUT_DIR_WITH_SUBFOLDER}/"eval.${EVAL_LABEL}.version_2.0.0.log.txt"


echo "Eval Label: ${EVAL_LABEL}"
echo "Possible label errors (sorted) file: "${MODEL_OUTPUT_DIR_WITH_SUBFOLDER}/"eval.${EVAL_LABEL}.possible_label_errors.jsonl"
echo "High reliablity region predictions (sorted) file: "${MODEL_OUTPUT_DIR_WITH_SUBFOLDER}/"eval.${EVAL_LABEL}.high_reliability.jsonl"
echo "All predictions file: "${MODEL_OUTPUT_DIR_WITH_SUBFOLDER}/"eval.${EVAL_LABEL}.all_predictions.jsonl"
echo "Eval log file: "${MODEL_OUTPUT_DIR_WITH_SUBFOLDER}/"eval.${EVAL_LABEL}.version_2.0.0.log.txt"

