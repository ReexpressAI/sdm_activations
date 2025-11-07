#########################################################################################################
##################### Setup
#########################################################################################################

# To use the following, copy `z_sdm_network_process_order_data.py` and/or
# `z_sdm_network_process_order_data_wikipedia.py` into the main reexpress/ folder.

# This will also construct datasets with multisets of size 4 and 5, but we only use the multisets of
# size 3 in the present experiments. Size 3 is already sufficiently challenging for the models, and
# has the benefit of having sufficiently low irreducible error when using exact string matches. With sets of
# size 5 and larger, the number of valid variations are such that one should consider using an exogenous model
# (e.g., the Reexpress MCP Server v2.0.0) for final scoring.

#########################################################################################################
##################### Harvard sentences data
#########################################################################################################

cd code/reexpress  # Update with the applicable path

conda activate re_mcp_v200  # conda environment with applicable dependencies

SUFFIX_SIZE=3

for SUFFIX_SIZE in "3" "4" "5"; do
echo "multiset size: ${SUFFIX_SIZE}"
OUTPUT_DATA_DIR="/Users/a/Documents/projects/sdm_paper/data/order/harvard_sentences_processed_suffix_size${SUFFIX_SIZE}"
mkdir -p ${OUTPUT_DATA_DIR}

python -u z_sdm_network_process_order_data.py \
--multiset_size ${SUFFIX_SIZE} \
--input_file="/Users/a/Documents/projects/sdm_paper/data/order/harvard_sentences/harvsents.txt" \
--output_genai_train_file="${OUTPUT_DATA_DIR}/train.jsonl" \
--output_genai_calibration_file="${OUTPUT_DATA_DIR}/calibration.jsonl" \
--output_held_out_eval_file="${OUTPUT_DATA_DIR}/test.jsonl" \
--output_all_combined_eval_file="${OUTPUT_DATA_DIR}/combined.jsonl"

done

#multiset size: 3
#Held-out eval set lines: 234
#Calibration set lines: 152
#Training set lines: 334
#Combined lines: 720
#Cumulative running time: 0.0200350284576416
#multiset size: 4
#Held-out eval set lines: 234
#Calibration set lines: 152
#Training set lines: 334
#Combined lines: 720
#Cumulative running time: 0.019912242889404297
#multiset size: 5
#Held-out eval set lines: 234
#Calibration set lines: 152
#Training set lines: 334
#Combined lines: 720
#Cumulative running time: 0.05463671684265137


#########################################################################################################
##################### Wikipedia sentences data
#########################################################################################################

cd code/reexpress  # Update with the applicable path

conda activate re_mcp_v200  # conda environment with applicable dependencies

for SUFFIX_SIZE in "3" "4" "5"; do
echo "multiset size: ${SUFFIX_SIZE}"
OUTPUT_DATA_DIR="/Users/a/Documents/projects/sdm_paper/data/order/wikipedia_sentences_processed_suffix_size${SUFFIX_SIZE}"
mkdir -p ${OUTPUT_DATA_DIR}

python -u z_sdm_network_process_order_data_wikipedia.py \
--multiset_size ${SUFFIX_SIZE} \
--output_genai_train_file="${OUTPUT_DATA_DIR}/train.jsonl" \
--output_genai_calibration_file="${OUTPUT_DATA_DIR}/calibration.jsonl" \
--output_held_out_eval_file="${OUTPUT_DATA_DIR}/test.jsonl" \
--output_held_out_challenge_eval_file="${OUTPUT_DATA_DIR}/challenge_test.jsonl" \
--output_remaining_lines_file="${OUTPUT_DATA_DIR}/remaining.jsonl"

done


#multiset size: 3
#Size of dataset after filtering by length: 7828653
#Lines: 7826653; min: 5; max: 60; mean: 19.108173187184867
#Longest subset: 2000; min: 60; max: 60; mean: 60.0
#Held-out challenge eval set lines (ordered by length): 2000
#Held-out eval set lines: 2000
#Calibration set lines: 5000
#Training set lines: 5000
#SKIPPING (insufficient variety): The Mark 2 was a fast and capable saloon in line with Sir William Lyons' 1950s advertising slogan: Grace . . .
#SKIPPING (insufficient variety): No one of them can assert a just claim to jurisdiction exclusively conferred on another, or withheld from all . . .
#SKIPPING (insufficient variety): During the 1990s, the proportion of black students in majority white schools has decreased . . .
#Remaining (unassigned) lines up to 100000: 99997
#Cumulative running time: 136.67100977897644
#multiset size: 4
#Size of dataset after filtering by length: 7828653
#Lines: 7826653; min: 5; max: 60; mean: 19.108173187184867
#Longest subset: 2000; min: 60; max: 60; mean: 60.0
#Held-out challenge eval set lines (ordered by length): 2000
#Held-out eval set lines: 2000
#Calibration set lines: 5000
#Training set lines: 5000
#Remaining (unassigned) lines up to 100000: 100000
#Cumulative running time: 143.6111810207367
#multiset size: 5
#Size of dataset after filtering by length: 7828653
#Lines: 7826653; min: 5; max: 60; mean: 19.108173187184867
#Longest subset: 2000; min: 60; max: 60; mean: 60.0
#Held-out challenge eval set lines (ordered by length): 2000
#Held-out eval set lines: 2000
#Calibration set lines: 5000
#Training set lines: 5000
#Remaining (unassigned) lines up to 100000: 100000
#Cumulative running time: 142.41557598114014

