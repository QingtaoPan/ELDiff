
set -e

# accelerate config
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8
NUM_GPUS=9
# MASTER_PORT=$(expr $RANDOM + 1000)
MASTER_PORT=$(shuf -i 10000-60000 -n 1)
# text config
TEXT_FILE_PATH="coco_obj_comp_5_1k.json"

# model config
MODEL_NAME="./EviDiff"

# output dir config
OUTPUT_DIR_NAME="MULTIGEN-result"
SAMPLE_IMG_DIR="sample_imgs/MULTIGEN-result"
SAMPLE_JSON_DIR="sample_jsons/MULTIGEN-result"

mkdir -p $SAMPLE_IMG_DIR
mkdir -p $SAMPLE_JSON_DIR


# gen images
#accelerate launch --num_processes $NUM_GPUS \
#    --main_process_port $MASTER_PORT \
#    gen_image_dist_multigpu.py \
#    --text_file_path $TEXT_FILE_PATH \
#    --model_name $MODEL_NAME \
#    --output_dir $SAMPLE_IMG_DIR

accelerate launch --num_processes 9 --main_process_port 29500 gen_image_dist_multigpu.py \
    --text_file_path $TEXT_FILE_PATH \
    --output_dir $SAMPLE_IMG_DIR \
    --model_name $MODEL_NAME

# gen jsons
accelerate launch --num_processes $NUM_GPUS \
    --main_process_port $MASTER_PORT \
    gen_json_dist.py \
    --text_file_path $TEXT_FILE_PATH \
    --image_dir $SAMPLE_IMG_DIR \
    --output_json_path $SAMPLE_JSON_DIR

# gen final multigen scores
python gen_multigen_scores.py \
    --text_file_path $TEXT_FILE_PATH \
    --result_path $SAMPLE_JSON_DIR \
    --name $OUTPUT_DIR_NAME 
