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
##################### SDM version
#########################################################################################################


cd code/reexpress  # Update with the applicable path

conda activate basetrain
export HF_HOME=/home/jupyter/models/hf

MODEL_LABEL_NAME="phi35_sdm_verification_layer"

DATA_DIR="/home/jupyter/data/order/wikipedia_sentences_processed_suffix_size3/"
RUN_ID="word_ordering_sdm_loss_gen0.5_suffix_size3_run2"
MODEL_DIR="/home/jupyter/models/order/preview__${MODEL_LABEL_NAME}_${RUN_ID}/models"
OUTPUT_DIR="/home/jupyter/models/order/preview__${MODEL_LABEL_NAME}_${RUN_ID}/outputs"


# Create directories
mkdir -p $DATA_DIR $MODEL_DIR $OUTPUT_DIR

echo "Data"
echo "  - Train: $DATA_DIR/train.jsonl ($(wc -l < $DATA_DIR/train.jsonl) examples)"
echo "  - Validation: $DATA_DIR/calibration.jsonl ($(wc -l < $DATA_DIR/calibration.jsonl) examples)"

# Step 2: Fine-tune the model
echo "Starting fine-tuning..."
echo ${OUTPUT_DIR}/run1.log.txt
mkdir -p $MODEL_DIR/${MODEL_LABEL_NAME}/generations
echo $MODEL_DIR/${MODEL_LABEL_NAME}/generations
mkdir -p $MODEL_DIR/${MODEL_LABEL_NAME}/verification_model
echo $MODEL_DIR/${MODEL_LABEL_NAME}/verification_model


# Verification layer parameters
ALPHA=0.95
EXEMPLAR_DIMENSION=1000
SDM_LEARNING_RATE=0.00001
SDM_EPOCHS=200

echo ${OUTPUT_DIR}/run1.log.txt
echo ${OUTPUT_DIR}/run1.err.txt

# Note that in this case since we are using 2 GPUs, we set --gradient_accumulation_steps 4 (2 still leads to OOM)

accelerate launch --num_processes=2 sdm_network_finetune.py \
    --generation_probability_during_training=0.5 \
    --train_file $DATA_DIR/train.jsonl \
    --eval_file $DATA_DIR/calibration.jsonl \
    --output_dir $MODEL_DIR/${MODEL_LABEL_NAME} \
    --generation_save_dir $MODEL_DIR/${MODEL_LABEL_NAME}/generations \
    --verification_model_dir=$MODEL_DIR/${MODEL_LABEL_NAME}/verification_model \
    --model_name "microsoft/Phi-3.5-mini-instruct" \
    --num_train_epochs 10 \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 8 \
    --gradient_accumulation_steps 4 \
    --learning_rate 5e-5 \
    --warmup_steps 78 \
    --logging_steps 39 \
    --save_steps 39 \
    --eval_steps 39 \
    --save_total_limit=2 \
    --bf16 \
    --gradient_checkpointing \
    --mask_prefix \
    --seed 0 \
    --max_generation_new_tokens 512 \
    --sdm_alpha=${ALPHA} \
    --sdm_class_size 2 \
    --sdm_epoch=${SDM_EPOCHS} \
    --sdm_batch_size 50 \
    --sdm_eval_batch_size 100 \
    --sdm_learning_rate ${SDM_LEARNING_RATE} \
    --sdm_number_of_random_shuffles 1 \
    --sdm_maxQAvailableFromIndexer 2048 \
    --sdm_exemplar_vector_dimension ${EXEMPLAR_DIMENSION} \
    --sdm_main_device="cuda:0" > ${OUTPUT_DIR}/run1.log.txt 2> ${OUTPUT_DIR}/run1.err.txt  # remember to pipe to standard error


# approximately 17.3 hours (62285.7219 seconds) on 2 gpus


python sdm_network_read_trainer_logs.py $MODEL_DIR/${MODEL_LABEL_NAME} \
--summary

    #{'eval_loss': 0.006277573294937611, 'eval_runtime': 1872.5327, 'eval_samples_per_second': 2.67, 'eval_steps_per_second': 0.167, 'epoch': 8.89}
    #
    #Evaluation Loss:
    #  Initial: 0.018766 (step 39)
    #  Final: 0.006931 (step 780)
    #  Best: 0.006278 (step 702)
    #  Number of evaluations: 20
  
python sdm_network_read_trainer_logs.py $MODEL_DIR/${MODEL_LABEL_NAME} \
--compare

 

# The following can be used for quick tests of prompts with the fine-tuned model and the base model:

python sdm_network_simple_chat.py $MODEL_DIR/${MODEL_LABEL_NAME} 0

python sdm_network_simple_chat.py "microsoft/Phi-3.5-mini-instruct" 0

# Also, if you want to try a specific checkpoint, you can use the following, for example using that of step 780:
CHECKPOINT="checkpoint-780"
python sdm_network_simple_chat.py ${MODEL_DIR}/${MODEL_LABEL_NAME}/${CHECKPOINT} 0

#########################################################################################################
##################### Decoding of the calibration set in preparation for training the final verification layer.
##################### In this case, we use the verification layer for the best evaluation step since
##################### the final has not yet been trained, but that
##################### is only for reference --- the SDM output in this case isn't used in downstream procecssing.
##################### Hard negatives derived from the calibration set are then used to train the final verification layer.
#########################################################################################################

cd code/reexpress  # Update with the applicable path

conda activate basetrain
export HF_HOME=/home/jupyter/models/hf

MODEL_LABEL_NAME="phi35_sdm_verification_layer"

RUN_ID="word_ordering_sdm_loss_gen0.5_suffix_size3_run2"
MODEL_DIR="/home/jupyter/models/order/preview__${MODEL_LABEL_NAME}_${RUN_ID}/models"
OUTPUT_DIR="/home/jupyter/models/order/preview__${MODEL_LABEL_NAME}_${RUN_ID}/outputs"


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
BEST_EVAL_STEP="eval_702"  # for reference analysis, we use the existing layer from the best evaluation, but this will be replaced with the final run below

OUTPUT_LOG_FILE=$OUTPUT_DIR/decode.${FILE_NAME}.${RUN_LABEL}.${BEST_EVAL_STEP}.log.txt
echo ${OUTPUT_LOG_FILE}



accelerate launch --num_processes=2 sdm_network_inference_with_verification.py \
    --line_limit=${LINE_LIMIT} \
    --model_path ${MODEL_DIR}/${MODEL_LABEL_NAME}/${CHECKPOINT} \
    --verification_model_dir=$MODEL_DIR/${MODEL_LABEL_NAME}/verification_model/${BEST_EVAL_STEP} \
    --input_file $DATA_DIR/${FILE_NAME}.jsonl \
    --output_file $OUTPUT_DIR/predictions.${FILE_NAME}.${RUN_LABEL}.${BEST_EVAL_STEP}.jsonl \
    --save_embedding \
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

#/home/jupyter/models/order/preview__phi35_sdm_verification_layer_word_ordering_sdm_loss_gen0.5_suffix_size3_run2/outputs/surface_level_eval.calibration.r1.suffix_size3.searchdepth0.temperature0.0.wikipedia.line_limit-1.eval_702.log.txt

    #Total examples: 5000
    #Verbatim exact matches (including formatting): 4657 (93.14%)
    #Exact matches (content only, first available tagged sentence): 4681 (93.62%)
    #No response: 0 (0.00%)
    #
    #Additional reference counts among parseable:
    #        verification_tp: 4657 (93.14%)
    #        verification_fp: 195 (3.90%)
    #        verification_fn: 24 (0.48%)
    #        verification_tn: 124 (2.48%)
    #
    #Word-level accuracy: 0.9809 ± 0.1152
    #Average attempts: 1.00 (max: 1)


# convert the file to the hard negatives format:

HARD_NEGATIVES_FILE=${EVAL_FILE}.processed_hard_negatives.jsonl

python -u sdm_network_process_generation_output_into_hard_negative_format.py \
--input_file=${EVAL_FILE} \
--output_hard_negatives_jsonl_file=${HARD_NEGATIVES_FILE}

#Documents with generated hard negatives: 319

# Train the final verification layer


CHECKPOINT=""
# Verification layer parameters
ALPHA=0.95
EXEMPLAR_DIMENSION=1000
SDM_LEARNING_RATE=0.00001
SDM_EPOCHS=200

echo ${OUTPUT_DIR}/verification_layer_final_training.run1.log.txt

## Train the verification layer from the best checkpoint.
# Note the use of --sdm_input_calibration_set_file instead of --sdm_input_training_set_file $DATA_DIR/train.jsonl \

accelerate launch --num_processes=2 sdm_network_train_verification_layer.py \
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

RUN_ID="word_ordering_sdm_loss_gen0.5_suffix_size3_run2"
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
    --model_path ${MODEL_DIR}/${MODEL_LABEL_NAME}/${CHECKPOINT} \
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

#/home/jupyter/models/order/preview__phi35_sdm_verification_layer_word_ordering_sdm_loss_gen0.5_suffix_size3_run2/outputs/surface_level_eval.combined.r1.suffix_size3.searchdepth0.temperature0.0.harvard.line_limit-1.log.txt

    #Total examples: 720
    #Verbatim exact matches (including formatting): 688 (95.56%)
    #Exact matches (content only, first available tagged sentence): 688 (95.56%)
    #No response: 0 (0.00%)
    #
    #Additional reference counts among parseable:
    #        verification_tp: 688 (95.56%)
    #        verification_fp: 19 (2.64%)
    #        verification_fn: 0 (0.00%)
    #        verification_tn: 13 (1.81%)
    #
    #Word-level accuracy: 0.9837 ± 0.0915
    #Average attempts: 1.00 (max: 1)

#/home/jupyter/models/order/preview__phi35_sdm_verification_layer_word_ordering_sdm_loss_gen0.5_suffix_size3_run2/outputs/surface_level_eval.test.r1.suffix_size3.searchdepth0.temperature0.0.wikipedia.line_limit-1.log.txt

    #Total examples: 2000
    #Verbatim exact matches (including formatting): 1859 (92.95%)
    #Exact matches (content only, first available tagged sentence): 1865 (93.25%)
    #No response: 0 (0.00%)
    #
    #Additional reference counts among parseable:
    #        verification_tp: 1859 (92.95%)
    #        verification_fp: 74 (3.70%)
    #        verification_fn: 6 (0.30%)
    #        verification_tn: 61 (3.05%)
    #
    #Word-level accuracy: 0.9801 ± 0.1154
    #Average attempts: 1.00 (max: 1)

#/home/jupyter/models/order/preview__phi35_sdm_verification_layer_word_ordering_sdm_loss_gen0.5_suffix_size3_run2/outputs/surface_level_eval.challenge_test.r1.suffix_size3.searchdepth0.temperature0.0.wikipedia.line_limit-1.log.txt

    #Total examples: 2000
    #Verbatim exact matches (including formatting): 1828 (91.40%)
    #Exact matches (content only, first available tagged sentence): 1839 (91.95%)
    #No response: 0 (0.00%)
    #
    #Additional reference counts among parseable:
    #        verification_tp: 1828 (91.40%)
    #        verification_fp: 107 (5.35%)
    #        verification_fn: 11 (0.55%)
    #        verification_tn: 54 (2.70%)
    #
    #Word-level accuracy: 0.9693 ± 0.1677
    #Average attempts: 1.00 (max: 1)

#########################################################################################################
##################### Eval of binary classification of instruction-following using the test-time SDM layer
#########################################################################################################

cd code/reexpress  # Update with the applicable path

conda activate basetrain
export HF_HOME=/home/jupyter/models/hf

MODEL_LABEL_NAME="phi35_sdm_verification_layer"

RUN_ID="word_ordering_sdm_loss_gen0.5_suffix_size3_run2"
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
