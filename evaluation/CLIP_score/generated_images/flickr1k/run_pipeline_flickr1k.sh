
set -e

# accelerate config
export CUDA_VISIBLE_DEVICES=0
NUM_GPUS=1
MASTER_PORT=$(expr $RANDOM + 1000)

# text config
TEXT_FILE_PATH="FlickrCaptions_1k.json"

# model config
MODEL_NAME="./EviDiff"

# output dir config
SAMPLE_IMG_DIR="sample_imgs/flickr1k/CLIP-result"

mkdir -p $SAMPLE_IMG_DIR

# gen images
accelerate launch --num_processes $NUM_GPUS \
    --main_process_port $MASTER_PORT \
    gen_image_dist_flickr1k.py \
    --text_file_path $TEXT_FILE_PATH \
    --model_name $MODEL_NAME \
    --output_dir $SAMPLE_IMG_DIR
