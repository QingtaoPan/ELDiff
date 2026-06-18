import torch
import json
import argparse
from tqdm import tqdm
from diffusers import StableDiffusionPipeline
from accelerate import PartialState

# from CLIP import clip
import clip
from torchvision import transforms
from PIL import Image
import numpy as np

device = "cuda:0"
model, preprocess = clip.load("ViT-B/32", device=device)


def compute_clip_text_sim(image: Image.Image, text: str):
    """
    计算 Text-Alignment (Text-sim)，即图像和文本之间的相似度
    """
    # 预处理图像
    image_input = preprocess(image).unsqueeze(0).to(device)
    # 编码文本
    text_input = clip.tokenize([text]).to(device)

    with torch.no_grad():
        image_features = model.encode_image(image_input)
        text_features = model.encode_text(text_input)

        # 特征归一化
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        # 计算余弦相似度
        similarity = (image_features @ text_features.T).item()

    return similarity

distributed_state = PartialState()

text_file_path = 'FlickrCaptions_1k.json'

with open(text_file_path, "r") as f:
    text_data = json.load(f)

# prepare the data
d = []
for index in range(len(text_data)):
    texts = [text_data[index]["text"] for _ in range(1)]
    for j in range(1):
        d.append((index, texts[j], j))
print("Data prepared.")

print("Start generating images.")
pp = 0
text_sim_sum = 0
with distributed_state.split_between_processes(d) as data:
    for index, text, j in tqdm(data):
        pp += 1
        img_id = "{}_{}".format(index, j)

        # calculate the CLIP Score of flickr1k
        generated_image = Image.open('/sample_imgs/flickr1k/CLIP-result/'
                                     +str(img_id)+'.png')

        # calculate the CLIP Score of coco5k
        # generated_image = Image.open('/sample_imgs/coco5k/CLIP-result/'
                                     # + str(img_id) + '.png')
        text_sim = compute_clip_text_sim(generated_image, text)
        text_sim_sum += text_sim
        text_sim_ave = text_sim_sum / pp
        print("Samples: {:.2f}, Object Accuracy: {:.4f}, Object Accuracy (ave): {:.4f}".format(pp, text_sim, text_sim_ave))

