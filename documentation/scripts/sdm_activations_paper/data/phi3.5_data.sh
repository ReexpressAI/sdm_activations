#########################################################################################################
##################### Overview
#########################################################################################################

# The original data files are available from their respective publications. We provide the links below.
# The add_phi_3_5_instruct_embeddings.py script can be used as a reference for adding new datasets
# with the expected input format.

#########################################################################################################
##################### FACTCHECK -- data source
#########################################################################################################

# The data is available in the link in the following publication, p. 5 (p. 971 of the Anthology). (Note that as of writing,
# the link in the PDF has an extraneous "https://aclanthology.org/" prefix in the URL.)

# Amos Azaria and Tom Mitchell. 2023. The Internal State of an LLM Knows When It's Lying. In Findings of the Association for Computational Linguistics: EMNLP 2023, pages 967–976, Singapore. Association for Computational Linguistics. https://doi.org/10.18653/v1/2023.findings-emnlp.68

# Convert the csv files into JSON lines files with the following fields:
#
# "id": unique id (e.g., as via str(uuid.uuid4()) in Python)
# "label": 0 or 1
# "document": the document text

# The rows of generated_true_false.csv are used for the held-out test set below labeled "ood_eval.jsonl".
# The rows of the remaining .csv files can be randomly assigned to "calibration.jsonl" and "train.jsonl".

# Recall from the SDM Activations paper that D_tr and D_ca are randomly split for each of J iterations of learning the SDM activation.
# We do the same for the adaptors of the baseline estimators. (The code performs the shuffling, so you can just deterministically
# concatenate the remaining files together at this stage to create two disjoint sets for "calibration.jsonl" and
# "train.jsonl". The key point is to ensure that the documents in the generated_true_false.csv file are
# assigned to the held-out set, which is named "ood_eval.jsonl" below.)

#########################################################################################################
##################### SENTIMENT -- data source
#########################################################################################################

# The in-distribution sentiment dataset is a subset of the commonly used benchmark data from the following publication:
#
# Andrew L. Maas, Raymond E. Daly, Peter T. Pham, Dan Huang, Andrew Y. Ng, and Christopher Potts. 2011. Learning Word Vectors for Sentiment Analysis. In Proceedings of the 49th Annual Meeting of the Association for Computational Linguistics: Human Language Technologies, pages 142–150, Portland, Oregon, USA. Association for Computational Linguistics. https://aclanthology.org/P11-1015/
#
# The original full dataset is here: https://ai.stanford.edu/~amaas/data/sentiment/
#
# For the in-distribution data, to simplify things, we recommend using the full original sets, or sampling smaller sets, as needed, to construct in-distribution training/calibration and held-out sets. The original splits used in the paper were constructed as follows, and were splits used from an earlier analysis of existing works. For training_set.jsonl, we used the 3.4k split from the following work:
#
# Kaushik, Divyansh, Eduard Hovy, and Zachary Lipton. 2020. Learning the difference that makes a difference with counterfactually-augmented data. In International Conference on Learning Representations.
#
# For calibration_set.jsonl, we used a disjoint 16k set of reviews from the original training set of Maas et al. (2011). The two held-out sets referenced below (the 488 documents of eval_set.jsonl and the 1583 documents of validation_set.jsonl) are remaining disjoint sets used for evaluation. The paper uses the larger 1583 set as D_te.
#
# The SemEval2017-task4-test.subtask-A.english.binaryevalformat.balanced.jsonl dataset ("SentimentOOD" in the paper) is from:
#
# Sara Rosenthal, Noura Farra, and Preslav Nakov. 2017. SemEval-2017 Task 4: Sentiment Analysis in Twitter. In Proceedings of the 11th International Workshop on Semantic Evaluation (SemEval-2017), pages 502–518, Vancouver, Canada. Association for Computational Linguistics. https://doi.org/10.18653/v1/S17-2088
#
# The balanced 4750-document SentimentOOD set can be constructed by using the following: https://github.com/allenschmaltz/exa/blob/main/replication_scripts/sentiment_data/data/semeval_2017_data_init.sh (which uses https://github.com/allenschmaltz/exa/blob/main/code/data/sentiment/counterfactual_dataset/semeval_2017/semeval_2017_task_4a_sentiment_data_to_binaryevalformat_v2.py).

# As with the Factcheck data, the inputs to the following scripts are JSON lines files. Convert the labeled documents into JSON lines files with the following fields:
#
# "id": unique id (e.g., as via str(uuid.uuid4()) in Python)
# "label": 0 or 1
# "document": the document text

#########################################################################################################
##################### FACTCHECK -- create shuffled test -- this is independent of the model
#########################################################################################################

cd code/data_processing/sdm_activations_paper

conda activate re_mcp_v200
export HF_HOME=/home/jupyter/models/hf


DATA_DIR="/home/jupyter/data/classification/factcheck"
INPUT_FILE_NAME="ood_eval"


python -u create_ood_shuffled_tests.py \
--input_file=${DATA_DIR}/${INPUT_FILE_NAME}.jsonl \
--dataset="factcheck" \
--output_file=${DATA_DIR}/${INPUT_FILE_NAME}.ood_random_shuffle.jsonl

echo ${DATA_DIR}/${INPUT_FILE_NAME}.ood_random_shuffle.jsonl

#########################################################################################################
##################### SENTIMENT -- create shuffled test -- this is independent of the model
#########################################################################################################

cd code/data_processing/sdm_activations_paper

conda activate re_mcp_v200
export HF_HOME=/home/jupyter/models/hf


DATA_DIR="/home/jupyter/data/classification/sentiment"

for INPUT_FILE_NAME in "eval_sets/SemEval2017-task4-test.subtask-A.english.binaryevalformat.balanced" "eval_sets/validation_set" "eval_sets/eval_set"; do
echo ${DATA_DIR}/${INPUT_FILE_NAME}.ood_random_shuffle.jsonl
python -u create_ood_shuffled_tests.py \
--input_file=${DATA_DIR}/${INPUT_FILE_NAME}.jsonl \
--output_file=${DATA_DIR}/${INPUT_FILE_NAME}.ood_random_shuffle.jsonl
done

#########################################################################################################
##################### FACTCHECK
#########################################################################################################

cd code/data_processing/sdm_activations_paper

conda activate re_mcp_v200
export HF_HOME=/home/jupyter/models/hf


DATA_DIR="/home/jupyter/data/classification/factcheck"

MODEL_LABEL="phi35"
OUTPUT_DIR="/home/jupyter/data/classification/factcheck_${MODEL_LABEL}"
mkdir -p ${OUTPUT_DIR}

for INPUT_FILE_NAME in "ood_eval" "calibration" "train"; do
echo ${INPUT_FILE_NAME}
python -u add_phi_3_5_instruct_embeddings.py \
--input_file=${DATA_DIR}/${INPUT_FILE_NAME}.jsonl \
--class_size=2 \
--dataset="factcheck" \
--main_device="cuda:0" \
--output_file=${OUTPUT_DIR}/${INPUT_FILE_NAME}.${MODEL_LABEL}.jsonl
done
#ood_eval
#Count of documents with embedding set to 0's: 0
#Phi-3.5-instruct accuracy       mean: 0.8326530612244898,       out of 245      (1.0)% of 245
#Class-conditional accuracy (i.e., stratified by TRUE class):
#Phi-3.5-instruct accuracy true class 0  mean: 0.9444444444444444,       out of 126      (0.5142857142857142)% of 245
#Phi-3.5-instruct accuracy true class 1  mean: 0.7142857142857143,       out of 119      (0.4857142857142857)% of 245
#Prediction-conditional accuracy (i.e., stratified by PREDICTED class):
#Phi-3.5-instruct accuracy predicted class 0     mean: 0.7777777777777778,       out of 153      (0.6244897959183674)% of 245
#Phi-3.5-instruct accuracy predicted class 1     mean: 0.9239130434782609,       out of 92       (0.37551020408163266)% of 245

#calibration
#Count of documents with embedding set to 0's: 0
#Phi-3.5-instruct accuracy       mean: 0.8468616496878081,       out of 3043     (1.0)% of 3043
#Class-conditional accuracy (i.e., stratified by TRUE class):
#Phi-3.5-instruct accuracy true class 0  mean: 0.9553035356904603,       out of 1499     (0.49260598093986196)% of 3043
#Phi-3.5-instruct accuracy true class 1  mean: 0.741580310880829,        out of 1544     (0.507394019060138)% of 3043
#Prediction-conditional accuracy (i.e., stratified by PREDICTED class):
#Phi-3.5-instruct accuracy predicted class 0     mean: 0.7820862916439104,       out of 1831     (0.6017088399605652)% of 3043
#Phi-3.5-instruct accuracy predicted class 1     mean: 0.9447194719471947,       out of 1212     (0.39829116003943477)% of 3043
#Cumulative running time: 265.06116557121277
#
#train
#Count of documents with embedding set to 0's: 0
#Phi-3.5-instruct accuracy       mean: 0.849112426035503,        out of 3042     (1.0)% of 3042
#Class-conditional accuracy (i.e., stratified by TRUE class):
#Phi-3.5-instruct accuracy true class 0  mean: 0.953227931488801,        out of 1518     (0.4990138067061144)% of 3042
#Phi-3.5-instruct accuracy true class 1  mean: 0.7454068241469817,       out of 1524     (0.5009861932938856)% of 3042
#Prediction-conditional accuracy (i.e., stratified by PREDICTED class):
#Phi-3.5-instruct accuracy predicted class 0     mean: 0.7885558583106267,       out of 1835     (0.6032215647600263)% of 3042
#Phi-3.5-instruct accuracy predicted class 1     mean: 0.9411764705882353,       out of 1207     (0.3967784352399737)% of 3042
#Cumulative running time: 254.63899683952332

# Shuffled data:
cd code/data_processing/sdm_activations_paper

conda activate re_mcp_v200
export HF_HOME=/home/jupyter/models/hf


DATA_DIR="/home/jupyter/data/classification/factcheck"
INPUT_FILE_NAME="ood_eval.ood_random_shuffle"

MODEL_LABEL="phi35"
OUTPUT_DIR="/home/jupyter/data/classification/factcheck_${MODEL_LABEL}"
mkdir -p ${OUTPUT_DIR}


echo ${INPUT_FILE_NAME}
python -u add_phi_3_5_instruct_embeddings.py \
--input_file=${DATA_DIR}/${INPUT_FILE_NAME}.jsonl \
--class_size=2 \
--dataset="factcheck" \
--main_device="cuda:0" \
--output_file=${OUTPUT_DIR}/${INPUT_FILE_NAME}.${MODEL_LABEL}.jsonl

#ood_eval.ood_random_shuffle
#Count of documents with embedding set to 0's: 0
#Phi-3.5-instruct accuracy       mean: 0.9142857142857143,       out of 245      (1.0)% of 245
#Class-conditional accuracy (i.e., stratified by TRUE class):
#Phi-3.5-instruct accuracy true class 0  mean: 0.9142857142857143,       out of 245      (1.0)% of 245
#Phi-3.5-instruct accuracy true class 1  mean: 0,        out of 0        (0.0)% of 245
#Prediction-conditional accuracy (i.e., stratified by PREDICTED class):
#Phi-3.5-instruct accuracy predicted class 0     mean: 1.0,      out of 224      (0.9142857142857143)% of 245
#Phi-3.5-instruct accuracy predicted class 1     mean: 0.0,      out of 21       (0.08571428571428572)% of 245
#Cumulative running time: 36.62098050117493



#########################################################################################################
##################### SENTIMENT
#########################################################################################################

cd code/data_processing/sdm_activations_paper

conda activate re_mcp_v200
export HF_HOME=/home/jupyter/models/hf


DATA_DIR="/home/jupyter/data/classification/sentiment"

MODEL_LABEL="phi35"
OUTPUT_DIR="/home/jupyter/data/classification/sentiment_${MODEL_LABEL}"
mkdir -p ${OUTPUT_DIR}/eval_sets

for INPUT_FILE_NAME in "eval_sets/SemEval2017-task4-test.subtask-A.english.binaryevalformat.balanced" "eval_sets/validation_set" "eval_sets/eval_set" "calibration_set" "training_set"; do
echo ${INPUT_FILE_NAME}
python -u add_phi_3_5_instruct_embeddings.py \
--input_file=${DATA_DIR}/${INPUT_FILE_NAME}.jsonl \
--class_size=2 \
--dataset="sentiment" \
--main_device="cuda:1" \
--output_file=${OUTPUT_DIR}/${INPUT_FILE_NAME}.${MODEL_LABEL}.jsonl
done


#eval_sets/SemEval2017-task4-test.subtask-A.english.binaryevalformat.balanced
#Count of documents with embedding set to 0's: 0
#Phi-3.5-instruct accuracy       mean: 0.7648421052631579,       out of 4750     (1.0)% of 4750
#Class-conditional accuracy (i.e., stratified by TRUE class):
#Phi-3.5-instruct accuracy true class 0  mean: 0.9962105263157894,       out of 2375     (0.5)% of 4750
#Phi-3.5-instruct accuracy true class 1  mean: 0.5334736842105263,       out of 2375     (0.5)% of 4750
#Prediction-conditional accuracy (i.e., stratified by PREDICTED class):
#Phi-3.5-instruct accuracy predicted class 0     mean: 0.6810592976396085,       out of 3474     (0.7313684210526316)% of 4750
#Phi-3.5-instruct accuracy predicted class 1     mean: 0.9929467084639498,       out of 1276     (0.26863157894736844)% of 4750
#Cumulative running time: 398.2669758796692
#
#eval_sets/validation_set
#Count of documents with embedding set to 0's: 0
#Phi-3.5-instruct accuracy       mean: 0.9147188881869868,       out of 1583     (1.0)% of 1583
#Class-conditional accuracy (i.e., stratified by TRUE class):
#Phi-3.5-instruct accuracy true class 0  mean: 0.9835858585858586,       out of 792      (0.5003158559696779)% of 1583
#Phi-3.5-instruct accuracy true class 1  mean: 0.8457648546144121,       out of 791      (0.4996841440303222)% of 1583
#Prediction-conditional accuracy (i.e., stratified by PREDICTED class):
#Phi-3.5-instruct accuracy predicted class 0     mean: 0.8645948945615982,       out of 901      (0.5691724573594441)% of 1583
#Phi-3.5-instruct accuracy predicted class 1     mean: 0.9809384164222874,       out of 682      (0.4308275426405559)% of 1583
#Cumulative running time: 139.58389854431152
#
#eval_sets/eval_set
#Count of documents with embedding set to 0's: 0
#Phi-3.5-instruct accuracy       mean: 0.9016393442622951,       out of 488      (1.0)% of 488
#Class-conditional accuracy (i.e., stratified by TRUE class):
#Phi-3.5-instruct accuracy true class 0  mean: 0.9753086419753086,       out of 243      (0.4979508196721312)% of 488
#Phi-3.5-instruct accuracy true class 1  mean: 0.8285714285714286,       out of 245      (0.5020491803278688)% of 488
#Prediction-conditional accuracy (i.e., stratified by PREDICTED class):
#Phi-3.5-instruct accuracy predicted class 0     mean: 0.8494623655913979,       out of 279      (0.5717213114754098)% of 488
#Phi-3.5-instruct accuracy predicted class 1     mean: 0.9712918660287081,       out of 209      (0.42827868852459017)% of 488
#Cumulative running time: 45.87763547897339
#
#calibration_set
#Count of documents with embedding set to 0's: 0
#Phi-3.5-instruct accuracy       mean: 0.9089824561403509,       out of 14250    (1.0)% of 14250
#Class-conditional accuracy (i.e., stratified by TRUE class):
#Phi-3.5-instruct accuracy true class 0  mean: 0.9848250667416046,       out of 7117     (0.49943859649122807)% of 14250
#Phi-3.5-instruct accuracy true class 1  mean: 0.8333099677555026,       out of 7133     (0.500561403508772)% of 14250
#Prediction-conditional accuracy (i.e., stratified by PREDICTED class):
#Phi-3.5-instruct accuracy predicted class 0     mean: 0.8549646255184191,       out of 8198     (0.5752982456140351)% of 14250
#Phi-3.5-instruct accuracy predicted class 1     mean: 0.9821546596166556,       out of 6052     (0.4247017543859649)% of 14250
#Cumulative running time: 1241.2058775424957
#
#training_set
#Count of documents with embedding set to 0's: 0
#Phi-3.5-instruct accuracy       mean: 0.9162272993555947,       out of 3414     (1.0)% of 3414
#Class-conditional accuracy (i.e., stratified by TRUE class):
#Phi-3.5-instruct accuracy true class 0  mean: 0.9801749271137026,       out of 1715     (0.5023432923257176)% of 3414
#Phi-3.5-instruct accuracy true class 1  mean: 0.85167745732784,         out of 1699     (0.4976567076742824)% of 3414
#Prediction-conditional accuracy (i.e., stratified by PREDICTED class):
#Phi-3.5-instruct accuracy predicted class 0     mean: 0.8696326952922918,       out of 1933     (0.5661980082015231)% of 3414
#Phi-3.5-instruct accuracy predicted class 1     mean: 0.9770425388251182,       out of 1481     (0.43380199179847684)% of 3414
#Cumulative running time: 292.5600788593292

# Shuffled data:
cd code/data_processing/sdm_activations_paper

conda activate re_mcp_v200
export HF_HOME=/home/jupyter/models/hf


DATA_DIR="/home/jupyter/data/classification/sentiment"

MODEL_LABEL="phi35"
OUTPUT_DIR="/home/jupyter/data/classification/sentiment_${MODEL_LABEL}"
mkdir -p ${OUTPUT_DIR}/eval_sets

for INPUT_FILE_NAME in "eval_sets/SemEval2017-task4-test.subtask-A.english.binaryevalformat.balanced.ood_random_shuffle" "eval_sets/validation_set.ood_random_shuffle" "eval_sets/eval_set.ood_random_shuffle"; do

echo ${INPUT_FILE_NAME}
python -u add_phi_3_5_instruct_embeddings.py \
--input_file=${DATA_DIR}/${INPUT_FILE_NAME}.jsonl \
--class_size=2 \
--dataset="sentiment" \
--main_device="cuda:0" \
--output_file=${OUTPUT_DIR}/${INPUT_FILE_NAME}.${MODEL_LABEL}.jsonl
done

#eval_sets/SemEval2017-task4-test.subtask-A.english.binaryevalformat.balanced.ood_random_shuffle
#Count of documents with embedding set to 0's: 0
#Phi-3.5-instruct accuracy       mean: 0.6726315789473685,       out of 4750     (1.0)% of 4750
#Class-conditional accuracy (i.e., stratified by TRUE class):
#Phi-3.5-instruct accuracy true class 0  mean: 0.9966315789473684,       out of 2375     (0.5)% of 4750
#Phi-3.5-instruct accuracy true class 1  mean: 0.3486315789473684,       out of 2375     (0.5)% of 4750
#Prediction-conditional accuracy (i.e., stratified by PREDICTED class):
#Phi-3.5-instruct accuracy predicted class 0     mean: 0.6047521716913643,       out of 3914     (0.824)% of 4750
#Phi-3.5-instruct accuracy predicted class 1     mean: 0.9904306220095693,       out of 836      (0.176)% of 4750
#Cumulative running time: 579.1765911579132
#eval_sets/validation_set.ood_random_shuffle
#Count of documents with embedding set to 0's: 0
#Phi-3.5-instruct accuracy       mean: 0.5862286797220467,       out of 1583     (1.0)% of 1583
#Class-conditional accuracy (i.e., stratified by TRUE class):
#Phi-3.5-instruct accuracy true class 0  mean: 0.9962121212121212,       out of 792      (0.5003158559696779)% of 1583
#Phi-3.5-instruct accuracy true class 1  mean: 0.17572692793931732,      out of 791      (0.4996841440303222)% of 1583
#Prediction-conditional accuracy (i.e., stratified by PREDICTED class):
#Phi-3.5-instruct accuracy predicted class 0     mean: 0.5475364330326162,       out of 1441     (0.9102969046114971)% of 1583
#Phi-3.5-instruct accuracy predicted class 1     mean: 0.9788732394366197,       out of 142      (0.08970309538850284)% of 1583
#Cumulative running time: 195.35255336761475
#eval_sets/eval_set.ood_random_shuffle
#Count of documents with embedding set to 0's: 0
#Phi-3.5-instruct accuracy       mean: 0.5922131147540983,       out of 488      (1.0)% of 488
#Class-conditional accuracy (i.e., stratified by TRUE class):
#Phi-3.5-instruct accuracy true class 0  mean: 0.9958847736625515,       out of 243      (0.4979508196721312)% of 488
#Phi-3.5-instruct accuracy true class 1  mean: 0.19183673469387755,      out of 245      (0.5020491803278688)% of 488
#Prediction-conditional accuracy (i.e., stratified by PREDICTED class):
#Phi-3.5-instruct accuracy predicted class 0     mean: 0.55,     out of 440      (0.9016393442622951)% of 488
#Phi-3.5-instruct accuracy predicted class 1     mean: 0.9791666666666666,       out of 48       (0.09836065573770492)% of 488
#Cumulative running time: 63.3652560710907
