#########################################################################################################
##################### Overview
#########################################################################################################

# This preprocesses the standard AGNews dataset available at https://huggingface.co/datasets/fancyzhx/ag_news.
# This is an auxiliary experiment to demonstrate that SDM estimators are not restricted to binary classification.

#########################################################################################################
##################### prepare data
#########################################################################################################

cd code/data_processing/sdm_activations_paper

conda activate re_mcp_v200

# We also need to install HuggingFace datasets
# We'll make a copy of the environment to avoid any unexpected conflicts with the existing environment for the main code:
conda create --name re_mcp_v200_with_datasets --clone re_mcp_v200

conda activate re_mcp_v200_with_datasets
# install datasets
pip install datasets==4.5.0


export HF_HOME=/home/jupyter/models/hf


OUTPUT_DIR="data/ag_news"  # Update with the applicable path

mkdir ${OUTPUT_DIR}

python -u ag_news_4class__step1_prepare_data.py \
--output_dir="${OUTPUT_DIR}"


#Lengths: Mean: 37.84745; min: 8; max: 177
#Total training documents: 120000
#Lengths: Mean: 37.72236842105263; min: 11; max: 137

# wc -l *
# lengths:
#   60000 data/ag_news/calibration.ag_news.jsonl
#    7600 data/ag_news/test.ag_news.jsonl
#   60000 data/ag_news/train.ag_news.jsonl
   
   
#########################################################################################################
##################### AGNEWS -- test
#########################################################################################################

cd code/data_processing/sdm_activations_paper

conda activate re_mcp_v200
export HF_HOME=/home/jupyter/models/hf


DATA_DIR="/home/jupyter/data/classification/ag_news" # Update with the applicable path

MODEL_LABEL="phi35"
OUTPUT_DIR="/home/jupyter/data/classification/ag_news_${MODEL_LABEL}"
mkdir -p ${OUTPUT_DIR}

##
INPUT_FILE_NAME="test.ag_news"

python -u add_phi_3_5_instruct_embeddings_agnews.py \
--input_file=${DATA_DIR}/${INPUT_FILE_NAME}.jsonl \
--class_size=4 \
--dataset="agnews" \
--main_device="cuda:0" \
--output_file=${OUTPUT_DIR}/${INPUT_FILE_NAME}.${MODEL_LABEL}.jsonl

##
#Count of documents with embedding set to 0's: 0
#Phi-3.5-instruct accuracy       mean: 0.8085526315789474,       out of 7600     (1.0)% of 7600
#Class-conditional accuracy (i.e., stratified by TRUE class):
#Phi-3.5-instruct accuracy true class 0  mean: 0.7905263157894736,       out of 1900     (0.25)% of 7600
#Phi-3.5-instruct accuracy true class 1  mean: 0.9847368421052631,       out of 1900     (0.25)% of 7600
#Phi-3.5-instruct accuracy true class 2  mean: 0.8994736842105263,       out of 1900     (0.25)% of 7600
#Phi-3.5-instruct accuracy true class 3  mean: 0.5594736842105263,       out of 1900     (0.25)% of 7600
#Prediction-conditional accuracy (i.e., stratified by PREDICTED class):
#Phi-3.5-instruct accuracy predicted class 0     mean: 0.9042745334136063,       out of 1661     (0.21855263157894736)% of 7600
#Phi-3.5-instruct accuracy predicted class 1     mean: 0.955079122001021,        out of 1959     (0.25776315789473686)% of 7600
#Phi-3.5-instruct accuracy predicted class 2     mean: 0.653787299158378,        out of 2614     (0.3439473684210526)% of 7600
#Phi-3.5-instruct accuracy predicted class 3     mean: 0.7781844802342606,       out of 1366     (0.17973684210526317)% of 7600
#Cumulative running time: 1101.7432935237885

#########################################################################################################
##################### AGNEWS -- train
#########################################################################################################

cd code/data_processing/sdm_activations_paper

conda activate re_mcp_v200
export HF_HOME=/home/jupyter/models/hf


DATA_DIR="/home/jupyter/data/classification/ag_news"

MODEL_LABEL="phi35"
OUTPUT_DIR="/home/jupyter/data/classification/ag_news_${MODEL_LABEL}"
mkdir -p ${OUTPUT_DIR}

INPUT_FILE_NAME="train.ag_news"

python -u add_phi_3_5_instruct_embeddings_agnews.py \
--input_file=${DATA_DIR}/${INPUT_FILE_NAME}.jsonl \
--class_size=4 \
--dataset="agnews" \
--main_device="cuda:1" \
--output_file=${OUTPUT_DIR}/${INPUT_FILE_NAME}.${MODEL_LABEL}.jsonl


#Count of documents with embedding set to 0's: 0
#Phi-3.5-instruct accuracy       mean: 0.8144166666666667,       out of 60000    (1.0)% of 60000
#Class-conditional accuracy (i.e., stratified by TRUE class):
#Phi-3.5-instruct accuracy true class 0  mean: 0.8035535368378866,       out of 14971    (0.24951666666666666)% of 60000
#Phi-3.5-instruct accuracy true class 1  mean: 0.9858958153150157,       out of 15031    (0.25051666666666667)% of 60000
#Phi-3.5-instruct accuracy true class 2  mean: 0.906870331740798,        out of 14861    (0.24768333333333334)% of 60000
#Phi-3.5-instruct accuracy true class 3  mean: 0.5641144216159081,       out of 15137    (0.25228333333333336)% of 60000
#Prediction-conditional accuracy (i.e., stratified by PREDICTED class):
#Phi-3.5-instruct accuracy predicted class 0     mean: 0.9187414082786008,       out of 13094    (0.21823333333333333)% of 60000
#Phi-3.5-instruct accuracy predicted class 1     mean: 0.9605263157894737,       out of 15428    (0.2571333333333333)% of 60000
#Phi-3.5-instruct accuracy predicted class 2     mean: 0.656102429287766,        out of 20541    (0.34235)% of 60000
#Phi-3.5-instruct accuracy predicted class 3     mean: 0.7807442625948615,       out of 10937    (0.18228333333333332)% of 60000
#Cumulative running time: 8349.297839164734


#########################################################################################################
##################### AGNEWS -- calibration
#########################################################################################################

cd code/data_processing/sdm_activations_paper

conda activate re_mcp_v200
export HF_HOME=/home/jupyter/models/hf


DATA_DIR="/home/jupyter/data/classification/ag_news"

MODEL_LABEL="phi35"
OUTPUT_DIR="/home/jupyter/data/classification/ag_news_${MODEL_LABEL}"
mkdir -p ${OUTPUT_DIR}

INPUT_FILE_NAME="calibration.ag_news"

python -u add_phi_3_5_instruct_embeddings_agnews.py \
--input_file=${DATA_DIR}/${INPUT_FILE_NAME}.jsonl \
--class_size=4 \
--dataset="agnews" \
--main_device="cuda:2" \
--output_file=${OUTPUT_DIR}/${INPUT_FILE_NAME}.${MODEL_LABEL}.jsonl


#Count of documents with embedding set to 0's: 0
#Phi-3.5-instruct accuracy       mean: 0.8159166666666666,       out of 60000    (1.0)% of 60000
#Class-conditional accuracy (i.e., stratified by TRUE class):
#Phi-3.5-instruct accuracy true class 0  mean: 0.7992547741034001,       out of 15029    (0.25048333333333334)% of 60000
#Phi-3.5-instruct accuracy true class 1  mean: 0.9833656222860578,       out of 14969    (0.24948333333333333)% of 60000
#Phi-3.5-instruct accuracy true class 2  mean: 0.908184160116256,        out of 15139    (0.2523166666666667)% of 60000
#Phi-3.5-instruct accuracy true class 3  mean: 0.5701406176411222,       out of 14863    (0.24771666666666667)% of 60000
#Prediction-conditional accuracy (i.e., stratified by PREDICTED class):
#Phi-3.5-instruct accuracy predicted class 0     mean: 0.9191919191919192,       out of 13068    (0.2178)% of 60000
#Phi-3.5-instruct accuracy predicted class 1     mean: 0.959145109793445,        out of 15347    (0.2557833333333333)% of 60000
#Phi-3.5-instruct accuracy predicted class 2     mean: 0.664459694567949,        out of 20692    (0.34486666666666665)% of 60000
#Phi-3.5-instruct accuracy predicted class 3     mean: 0.777930781235656,        out of 10893    (0.18155)% of 60000
#Cumulative running time: 8403.994407653809
