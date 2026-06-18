
set -e

# accelerate config
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8
NUM_GPUS=9
MASTER_PORT=$(shuf -i 10000-60000 -n 1)

# text config
TEXT_FILE_PATH="coco_val2017captions_5k.json"

# model config
MODEL_NAME="./EviDiff"

# output dir config
SAMPLE_IMG_DIR="sample_imgs/coco5k/CLIP-result"

mkdir -p $SAMPLE_IMG_DIR

# gen images
accelerate launch --num_processes $NUM_GPUS \
    --main_process_port $MASTER_PORT \
    gen_image_dist_coco5k.py \
    --text_file_path $TEXT_FILE_PATH \
    --model_name $MODEL_NAME \
    --output_dir $SAMPLE_IMG_DIR
