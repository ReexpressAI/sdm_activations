#########################################################################################################
##################### Overview
#########################################################################################################

# The preprocessed data is available in the GitHub release, but we include this for reference.
# The add_mixtral_8x7b_instruct_embeddings.py script can be used as a reference for adding new datasets
# with the expected input format. (The shuffling of the data for the OOD datasets occurs once,
# and the approach for doing so is in the Phi3.5 script.)

#########################################################################################################
##################### FACTCHECK
#########################################################################################################

cd code/data_processing/sdm_activations_paper

conda activate re_mcp_v200
export HF_HOME=/home/jupyter/models/hf
#hf auth login

DATA_DIR="/home/jupyter/data/classification/factcheck"

MODEL_LABEL="mixtral_8x7b"
OUTPUT_DIR="/home/jupyter/data/classification/factcheck_${MODEL_LABEL}"
mkdir -p ${OUTPUT_DIR}

for INPUT_FILE_NAME in "ood_eval" "calibration" "train"; do
echo ${INPUT_FILE_NAME}
python -u add_mixtral_8x7b_instruct_embeddings.py \
--input_file=${DATA_DIR}/${INPUT_FILE_NAME}.jsonl \
--class_size=2 \
--dataset="factcheck" \
--output_file=${OUTPUT_DIR}/${INPUT_FILE_NAME}.${MODEL_LABEL}.jsonl
done


#/home/jupyter/data/classification/factcheck_mixtral_8x7b
#
#Count of documents with embedding set to 0's: 0
#Mixtral-8x7B-Instruct-v0.1 accuracy     mean: 0.7346938775510204,       out of 245      (1.0)% of 245
#Class-conditional accuracy (i.e., stratified by TRUE class):
#Mixtral-8x7B-Instruct-v0.1 accuracy true class 0        mean: 0.9761904761904762,       out of 126      (0.5142857142857142)% of 245
#Mixtral-8x7B-Instruct-v0.1 accuracy true class 1        mean: 0.4789915966386555,       out of 119      (0.4857142857142857)% of 245
#Prediction-conditional accuracy (i.e., stratified by PREDICTED class):
#Mixtral-8x7B-Instruct-v0.1 accuracy predicted class 0   mean: 0.6648648648648648,       out of 185      (0.7551020408163265)% of 245
#Mixtral-8x7B-Instruct-v0.1 accuracy predicted class 1   mean: 0.95,     out of 60       (0.24489795918367346)% of 245


#Count of documents with embedding set to 0's: 0                                                                               
#Mixtral-8x7B-Instruct-v0.1 accuracy     mean: 0.7860663818600065,       out of 3043     (1.0)% of 3043
#Class-conditional accuracy (i.e., stratified by TRUE class):                                                                  
#Mixtral-8x7B-Instruct-v0.1 accuracy true class 0        mean: 0.981320880587058,        out of 1499     (0.49260598093986196)% of 3043
#Mixtral-8x7B-Instruct-v0.1 accuracy true class 1        mean: 0.5965025906735751,       out of 1544     (0.507394019060138)% of 3043
#Prediction-conditional accuracy (i.e., stratified by PREDICTED class):
#Mixtral-8x7B-Instruct-v0.1 accuracy predicted class 0   mean: 0.7024832855778415,       out of 2094     (0.6881367071968452)% of 3043
#Mixtral-8x7B-Instruct-v0.1 accuracy predicted class 1   mean: 0.9704952581664911,       out of 949      (0.3118632928031548)% of 3043
#Cumulative running time: 845.4474196434021

#Count of documents with embedding set to 0's: 0
#Mixtral-8x7B-Instruct-v0.1 accuracy     mean: 0.7915844838921762,       out of 3042     (1.0)% of 3042
#Class-conditional accuracy (i.e., stratified by TRUE class):
#Mixtral-8x7B-Instruct-v0.1 accuracy true class 0        mean: 0.9749670619235836,       out of 1518     (0.4990138067061144)% of 3042
#Mixtral-8x7B-Instruct-v0.1 accuracy true class 1        mean: 0.6089238845144357,       out of 1524     (0.5009861932938856)% of 3042
#Prediction-conditional accuracy (i.e., stratified by PREDICTED class):
#Mixtral-8x7B-Instruct-v0.1 accuracy predicted class 0   mean: 0.7129094412331407,       out of 2076     (0.6824457593688363)% of 3042
#Mixtral-8x7B-Instruct-v0.1 accuracy predicted class 1   mean: 0.9606625258799172,       out of 966      (0.3175542406311637)% of 3042
#Cumulative running time: 826.5555377006531

# Shuffled data:
cd code/data_processing/sdm_activations_paper

conda activate re_mcp_v200
export HF_HOME=/home/jupyter/models/hf
#hf auth login

DATA_DIR="/home/jupyter/data/classification/factcheck"
INPUT_FILE_NAME="ood_eval.ood_random_shuffle"

MODEL_LABEL="mixtral_8x7b"
OUTPUT_DIR="/home/jupyter/data/classification/factcheck_${MODEL_LABEL}"
mkdir -p ${OUTPUT_DIR}


echo ${OUTPUT_DIR}/${INPUT_FILE_NAME}.${MODEL_LABEL}.jsonl
python -u add_mixtral_8x7b_instruct_embeddings.py \
--input_file=${DATA_DIR}/${INPUT_FILE_NAME}.jsonl \
--class_size=2 \
--dataset="factcheck" \
--output_file=${OUTPUT_DIR}/${INPUT_FILE_NAME}.${MODEL_LABEL}.jsonl

#/home/jupyter/data/classification/factcheck_mixtral_8x7b/ood_eval.ood_random_shuffle.mixtral_8x7b.jsonl

#Count of documents with embedding set to 0's: 0
#Mixtral-8x7B-Instruct-v0.1 accuracy     mean: 0.9755102040816327,       out of 245      (1.0)% of 245
#Class-conditional accuracy (i.e., stratified by TRUE class):
#Mixtral-8x7B-Instruct-v0.1 accuracy true class 0        mean: 0.9755102040816327,       out of 245      (1.0)% of 245
#Mixtral-8x7B-Instruct-v0.1 accuracy true class 1        mean: 0,        out of 0        (0.0)% of 245
#Prediction-conditional accuracy (i.e., stratified by PREDICTED class):
#Mixtral-8x7B-Instruct-v0.1 accuracy predicted class 0   mean: 1.0,      out of 239      (0.9755102040816327)% of 245
#Mixtral-8x7B-Instruct-v0.1 accuracy predicted class 1   mean: 0.0,      out of 6        (0.024489795918367346)% of 245
#Cumulative running time: 126.09810543060303

#########################################################################################################
##################### SENTIMENT
#########################################################################################################

cd code/data_processing/sdm_activations_paper

conda activate re_mcp_v200
export HF_HOME=/home/jupyter/models/hf
#hf auth login

DATA_DIR="/home/jupyter/data/classification/sentiment"

MODEL_LABEL="mixtral_8x7b"
OUTPUT_DIR="/home/jupyter/data/classification/sentiment_${MODEL_LABEL}"
mkdir -p ${OUTPUT_DIR}/eval_sets

for INPUT_FILE_NAME in "eval_sets/SemEval2017-task4-test.subtask-A.english.binaryevalformat.balanced" "eval_sets/validation_set" "eval_sets/eval_set" "calibration_set" "training_set"; do
echo ${INPUT_FILE_NAME}
echo ${OUTPUT_DIR}/${INPUT_FILE_NAME}.${MODEL_LABEL}.log.txt
python -u add_mixtral_8x7b_instruct_embeddings.py \
--input_file=${DATA_DIR}/${INPUT_FILE_NAME}.jsonl \
--class_size=2 \
--dataset="sentiment" \
--output_file=${OUTPUT_DIR}/${INPUT_FILE_NAME}.${MODEL_LABEL}.jsonl > ${OUTPUT_DIR}/${INPUT_FILE_NAME}.${MODEL_LABEL}.log.txt
done


#/home/jupyter/data/classification/sentiment_mixtral_8x7b/eval_sets/SemEval2017-task4-test.subtask-A.english.binaryevalformat.balanced.mixtral_8x7b.log.txt
#"eval_sets/SemEval2017-task4-test.subtask-A.english.binaryevalformat.balanced"
#Currently processing instance 0
#Count of documents with embedding set to 0's: 0
#Mixtral-8x7B-Instruct-v0.1 accuracy     mean: 0.6741052631578948,       out of 4750     (1.0)% of 4750
#Class-conditional accuracy (i.e., stratified by TRUE class):
#Mixtral-8x7B-Instruct-v0.1 accuracy true class 0        mean: 0.9983157894736842,       out of 2375     (0.5)% of 4750
#Mixtral-8x7B-Instruct-v0.1 accuracy true class 1        mean: 0.34989473684210526,      out of 2375     (0.5)% of 4750
#Prediction-conditional accuracy (i.e., stratified by PREDICTED class):
#Mixtral-8x7B-Instruct-v0.1 accuracy predicted class 0   mean: 0.6056194125159643,       out of 3915     (0.8242105263157895)% of 4750
#Mixtral-8x7B-Instruct-v0.1 accuracy predicted class 1   mean: 0.9952095808383233,       out of 835      (0.17578947368421052)% of 4750
#Cumulative running time: 1301.2458906173706
#
#"eval_sets/validation_set"
#Currently processing instance 0
#Count of documents with embedding set to 0's: 0
#Mixtral-8x7B-Instruct-v0.1 accuracy     mean: 0.9273531269740998,       out of 1583     (1.0)% of 1583
#Class-conditional accuracy (i.e., stratified by TRUE class):
#Mixtral-8x7B-Instruct-v0.1 accuracy true class 0        mean: 0.9785353535353535,       out of 792      (0.5003158559696779)% of 1583
#Mixtral-8x7B-Instruct-v0.1 accuracy true class 1        mean: 0.8761061946902655,       out of 791      (0.4996841440303222)% of 1583
#Prediction-conditional accuracy (i.e., stratified by PREDICTED class):
#Mixtral-8x7B-Instruct-v0.1 accuracy predicted class 0   mean: 0.8877434135166093,       out of 873      (0.5514845230574857)% of 1583
#Mixtral-8x7B-Instruct-v0.1 accuracy predicted class 1   mean: 0.976056338028169,        out of 710      (0.4485154769425142)% of 1583
#Cumulative running time: 494.2790298461914
#
#"eval_sets/eval_set"
#Currently processing instance 0
#Count of documents with embedding set to 0's: 0
#Mixtral-8x7B-Instruct-v0.1 accuracy     mean: 0.9200819672131147,       out of 488      (1.0)% of 488
#Class-conditional accuracy (i.e., stratified by TRUE class):
#Mixtral-8x7B-Instruct-v0.1 accuracy true class 0        mean: 0.9711934156378601,       out of 243      (0.4979508196721312)% of 488
#Mixtral-8x7B-Instruct-v0.1 accuracy true class 1        mean: 0.8693877551020408,       out of 245      (0.5020491803278688)% of 488
#Prediction-conditional accuracy (i.e., stratified by PREDICTED class):
#Mixtral-8x7B-Instruct-v0.1 accuracy predicted class 0   mean: 0.8805970149253731,       out of 268      (0.5491803278688525)% of 488
#Mixtral-8x7B-Instruct-v0.1 accuracy predicted class 1   mean: 0.9681818181818181,       out of 220      (0.45081967213114754)% of 488
#Cumulative running time: 180.8283712863922

#"calibration_set"
#Currently processing instance 0
#Count of documents with embedding set to 0's: 0
#Mixtral-8x7B-Instruct-v0.1 accuracy     mean: 0.9331929824561404,       out of 14250    (1.0)% of 14250
#Class-conditional accuracy (i.e., stratified by TRUE class):
#Mixtral-8x7B-Instruct-v0.1 accuracy true class 0        mean: 0.9808908247857243,       out of 7117     (0.49943859649122807)% of 14250
#Mixtral-8x7B-Instruct-v0.1 accuracy true class 1        mean: 0.8856021309406982,       out of 7133     (0.500561403508772)% of 14250
#Prediction-conditional accuracy (i.e., stratified by PREDICTED class):
#Mixtral-8x7B-Instruct-v0.1 accuracy predicted class 0   mean: 0.8953443632166218,       out of 7797     (0.5471578947368421)% of 14250
#Mixtral-8x7B-Instruct-v0.1 accuracy predicted class 1   mean: 0.9789245312257865,       out of 6453     (0.4528421052631579)% of 14250
#Cumulative running time: 4221.404235363007
#
#"training_set"
#Currently processing instance 0
#Count of documents with embedding set to 0's: 0
#Mixtral-8x7B-Instruct-v0.1 accuracy     mean: 0.9349736379613357,       out of 3414     (1.0)% of 3414
#Class-conditional accuracy (i.e., stratified by TRUE class):
#Mixtral-8x7B-Instruct-v0.1 accuracy true class 0        mean: 0.9760932944606414,       out of 1715     (0.5023432923257176)% of 3414
#Mixtral-8x7B-Instruct-v0.1 accuracy true class 1        mean: 0.8934667451442024,       out of 1699     (0.4976567076742824)% of 3414
#Prediction-conditional accuracy (i.e., stratified by PREDICTED class):
#Mixtral-8x7B-Instruct-v0.1 accuracy predicted class 0   mean: 0.9024258760107817,       out of 1855     (0.5433509080257762)% of 3414
#Mixtral-8x7B-Instruct-v0.1 accuracy predicted class 1   mean: 0.9737010904425915,       out of 1559     (0.45664909197422376)% of 3414
#Cumulative running time: 1138.024906873703

# Shuffled data
cd code/data_processing/sdm_activations_paper

conda activate re_mcp_v200
export HF_HOME=/home/jupyter/models/hf
#hf auth login

DATA_DIR="/home/jupyter/data/classification/sentiment"

MODEL_LABEL="mixtral_8x7b"
OUTPUT_DIR="/home/jupyter/data/classification/sentiment_${MODEL_LABEL}"
mkdir -p ${OUTPUT_DIR}/eval_sets

for INPUT_FILE_NAME in "eval_sets/SemEval2017-task4-test.subtask-A.english.binaryevalformat.balanced.ood_random_shuffle" "eval_sets/validation_set.ood_random_shuffle" "eval_sets/eval_set.ood_random_shuffle"; do
echo ${INPUT_FILE_NAME}
echo ${OUTPUT_DIR}/${INPUT_FILE_NAME}.${MODEL_LABEL}.log.txt
python -u add_mixtral_8x7b_instruct_embeddings.py \
--input_file=${DATA_DIR}/${INPUT_FILE_NAME}.jsonl \
--class_size=2 \
--dataset="sentiment" \
--output_file=${OUTPUT_DIR}/${INPUT_FILE_NAME}.${MODEL_LABEL}.jsonl > ${OUTPUT_DIR}/${INPUT_FILE_NAME}.${MODEL_LABEL}.log.txt
done


#"eval_sets/SemEval2017-task4-test.subtask-A.english.binaryevalformat.balanced.ood_random_shuffle"
#Currently processing instance 0
#Count of documents with embedding set to 0's: 0
#Mixtral-8x7B-Instruct-v0.1 accuracy     mean: 0.5621052631578948,       out of 4750     (1.0)% of 4750
#Class-conditional accuracy (i.e., stratified by TRUE class):
#Mixtral-8x7B-Instruct-v0.1 accuracy true class 0        mean: 0.999578947368421,        out of 2375     (0.5)% of 4750
#Mixtral-8x7B-Instruct-v0.1 accuracy true class 1        mean: 0.12463157894736843,      out of 2375     (0.5)% of 4750
#Prediction-conditional accuracy (i.e., stratified by PREDICTED class):
#Mixtral-8x7B-Instruct-v0.1 accuracy predicted class 0   mean: 0.5331237368066472,       out of 4453     (0.9374736842105263)% of 4750
#Mixtral-8x7B-Instruct-v0.1 accuracy predicted class 1   mean: 0.9966329966329966,       out of 297      (0.06252631578947368)% of 4750
#Cumulative running time: 1465.449114561081

#"eval_sets/validation_set.ood_random_shuffle"
#Currently processing instance 0
#Count of documents with embedding set to 0's: 0
#Mixtral-8x7B-Instruct-v0.1 accuracy     mean: 0.6696146557169931,       out of 1583     (1.0)% of 1583
#Class-conditional accuracy (i.e., stratified by TRUE class):
#Mixtral-8x7B-Instruct-v0.1 accuracy true class 0        mean: 0.98989898989899,         out of 792      (0.5003158559696779)% of 1583
#Mixtral-8x7B-Instruct-v0.1 accuracy true class 1        mean: 0.34892541087231355,      out of 791      (0.4996841440303222)% of 1583
#Prediction-conditional accuracy (i.e., stratified by PREDICTED class):
#Mixtral-8x7B-Instruct-v0.1 accuracy predicted class 0   mean: 0.6035411855273287,       out of 1299     (0.8205938092229943)% of 1583
#Mixtral-8x7B-Instruct-v0.1 accuracy predicted class 1   mean: 0.971830985915493,        out of 284      (0.17940619077700568)% of 1583
#Cumulative running time: 574.3991160392761

#"eval_sets/eval_set.ood_random_shuffle"
#Currently processing instance 0
#Count of documents with embedding set to 0's: 0
#Mixtral-8x7B-Instruct-v0.1 accuracy     mean: 0.6639344262295082,       out of 488      (1.0)% of 488
#Class-conditional accuracy (i.e., stratified by TRUE class):
#Mixtral-8x7B-Instruct-v0.1 accuracy true class 0        mean: 1.0,      out of 243      (0.4979508196721312)% of 488
#Mixtral-8x7B-Instruct-v0.1 accuracy true class 1        mean: 0.3306122448979592,       out of 245      (0.5020491803278688)% of 488
#Prediction-conditional accuracy (i.e., stratified by PREDICTED class):
#Mixtral-8x7B-Instruct-v0.1 accuracy predicted class 0   mean: 0.597051597051597,        out of 407      (0.8340163934426229)% of 488
#Mixtral-8x7B-Instruct-v0.1 accuracy predicted class 1   mean: 1.0,      out of 81       (0.16598360655737704)% of 488
#Cumulative running time: 213.95315027236938

