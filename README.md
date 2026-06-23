<div align="center">
  
# ELDiff: When Evidential Learning Meets Text-to-Image Diffusion

[Qingtao Pan](https://qingtaopan.github.io/), [Kai Ye](https://scholar.google.com/citations?hl=zh-CN&user=k6mAT9AAAAAJ), [Zhihao Dou](https://scholar.google.com/citations?user=JiBGiB8AAAAJ&hl=zh-CN&oi=ao), [Bing Ji](https://www.researchgate.net/profile/Bing-Ji-jibing), and [Shuo Li](https://case.edu/engineering/about/faculty-and-staff-directory/shuo-li)

**ECCV, 2026**

<a href='https://arxiv.org/abs/2606.20924'><img src='https://img.shields.io/badge/Paper-Arxiv-red'></a>

</div>

## Overview
In this paper, we propose ELDiff, an evidential learning-supervised T2I diffusion model, which leverages the advantages of uncertainty metric and conflict detection to enhance the fault tolerance of unreliable segmentation maps and suppress semantic conflicts, strengthening object-wise consistency learning. Specifically, a pixel evidence loss is proposed to restrain overconfidence in unreliable labels through evidential regularization, and a token conflict loss is designed to weaken the contradiction between semantics through optimizing a measured conflict factor.
<p align="center">
  <img src="figs/fig2.jpg" width="70%"></a> <br>
</p>

## 🆕 Models

| Stable Diffusion Version | Checkpoint |
|:------------------------:|:------------:|
| v1.4                     | [ELDiff_SD14](https://huggingface.co/pqt33/ELDiff_SD14)    |
| v2.1                     | [ELDiff_SD21](https://huggingface.co/pqt33/ELDiff_SD21)    

You can use the following code to download our checkpoints and generate images:
```python
import torch
from diffusers import StableDiffusionPipeline

model_id = "./ELDiff_SD14"
device = "cuda"

pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float32)
pipe = pipe.to(device)

prompt = "A cheese cat is sucking catnip"
image = pipe(prompt).images[0]  
    
image.save("cat.png")
```

## 🌏 Environment Setup
Create and activate the environment:
```bash
conda create -n ELDiff python=3.8.5
conda activate ELDiff
conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 pytorch-cuda=11.7 -c pytorch -c nvidia
pip install -r requirements.txt
```

## 🗂️ Fine-tuning Data Setup
### 1. Setup the COCO Image Data

```bash
cd train/data
# download COCO train2017
wget http://images.cocodataset.org/zips/train2017.zip
unzip train2017.zip
rm train2017.zip
bash coco_data_setup.sh
```

After this step, you should have the following structure under the `train/data`  directory:

```
train/data/
    coco_gsam_img/
        train/
            000000000142.jpg
            000000000370.jpg
            ...
```

### 2. Setup Token-wise Grounded Segmentation Maps

Download COCO segmentation data from [coco_gsam_seg](https://huggingface.co/datasets/pqt33/coco_gsam_seg) and put it under `train/data` directory.

After this step, you should have the following structure under the `train/data` directory:

```
train/data/
    coco_gsam_img/
        train/
            000000000142.jpg
            000000000370.jpg
            ...
    coco_gsam_seg.tar
```

Then, run the following command to unzip the segmentation data:

```bash
cd train/data
tar -xvf coco_gsam_seg.tar
rm coco_gsam_seg.tar
```

After the setup, you should have the following structure under the `train/data` directory:

```
train/data/
    coco_gsam_img/
        train/
            000000000142.jpg
            000000000370.jpg
            ...
    coco_gsam_seg/
        000000000142/
            mask_000000000142_bananas.png
            mask_000000000142_bread.png
            ...
        000000000370/
            mask_000000000370_bananas.png
            mask_000000000370_bread.png
            ...
        ...
```

## 🔥 Fine-tuning 
use the following command:
```bash
cd train
bash train.sh
```
The results will be saved under `train/results` directory.

## 📈 Evaluation
### 1. VISOR
First, generate images for VISOR
```bash
cd evaluation/VISOR/generate_images
bash run_pipeline_VISOR.sh
```
Second, calculate OA score
```bash
cd evaluation/VISOR/calculate_OA_score
python gen_img_obj.py
```

### 2. MULTIGEN
```bash
cd evaluation/MULTIGEN
bash run_pipeline.sh
```

### 3. CLIP_score of coco5k
First, generate images for coco5k
```bash
cd evaluation/CLIP_score/generated_images/coco5k
bash run_pipeline_coco5k_multigpu.sh
```
Second, calculate CLIP_score for coco5k
```bash
cd evaluation/CLIP_score/calculate_CLIP_Score
python text-image-similarity.py
```

### 4. CLIP_score of flickr1k
First, generate images for flickr1k
```bash
cd evaluation/CLIP_score/generated_images/flickr1k
bash run_pipeline_flickr1k.sh
```
Second, calculate CLIP_score for flickr1k
```bash
cd evaluation/CLIP_score/calculate_CLIP_Score
# modify the path: text_file_path = 'FlickrCaptions_1k.json' and generated_image = Image.open('/sample_imgs/flickr1k/CLIP-result/'+str(img_id)+'.png')
python text-image-similarity.py
```

### 5. FID of coco5k and flickr1k
For FID calculation, we use the generated images from CLIP_score of coco5k and CLIP_score of flickr1k
```bash
pip install pytorch-fid
# calculate FID of coco5k
python -m pytorch_fid --batch-size=1 ./sample_imgs/coco5k/CLIP-result ./data/coco5k
# calculate FID of flickr1k
python -m pytorch_fid --batch-size=1 ./sample_imgs/flickr1k/CLIP-result ./data/flickr30k
```

### 6. Other evaluations
For other evaluations such as T2I-CompBench, T2I-CompBench++, GenEval, and GenEval2, please follow their official code.


## 📜 License

This repository is released under the [Apache 2.0](LICENSE) license. 

## 🙏 Acknowledgement

Our code is built upon [diffusers](https://github.com/huggingface/diffusers), [prompt-to-prompt](https://github.com/google/prompt-to-prompt), [VISOR](https://github.com/microsoft/VISOR), [Grounded-Segment-Anything](https://github.com/IDEA-Research/Grounded-Segment-Anything), [CLIP](https://github.com/openai/CLIP), and [TokenCompose](https://github.com/mlpc-ucsd/TokenCompose). We thank all these authors for their nicely open sourced code and their great contributions to the community.

## 📚 Citation

If you find our work useful, please consider citing:
```bibtex
@misc{pan2026eldiffevidentiallearningmeets,
      title={ELDiff: When Evidential Learning Meets Text-to-Image Diffusion}, 
      author={Qingtao Pan and Kai Ye and Zhihao Dou and Bing Ji and Shuo Li},
      journal={arXiv preprint arXiv:2606.20924},
      year={2026}
}
```





