import os, json, itertools
from PIL import Image
import requests
import numpy as np
import random
from PIL import Image

import numpy as np
import itertools

# object detector model
import transformers
import torch
from transformers import OwlViTProcessor, OwlViTForObjectDetection
import numpy as np

processor = OwlViTProcessor.from_pretrained("google/owlvit-base-patch32")
model = OwlViTForObjectDetection.from_pretrained("google/owlvit-base-patch32")
device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)
model.eval()
# %%
MODELS = ["stable-diffusion"]
URL_PREFIX = 'sample_imgs/VISOR-result'

viz_ids = [
    19100, 24653, 29131, 8606, 17652, 6603, 26515, 22815, 7904, 6486, 26363,
    22495, 18253, 12812, 20714, 29841, 23283, 29120, 23113, 810, 9942, 22356,
    3792, 7257, 29971, 20086, 20727, 10321, 2084, 27141, 30955, 29633, 23544,
    13352, 27244, 19973, 7646, 21186, 7366, 17831, 8001, 12373, 12046, 8966,
    7264, 15896, 29727, 5257, 4254, 8754, 17066, 7170, 26186, 16226, 8341,
    10516, 25814, 887, 19792, 24514, 3937, 27667, 19794, 7335, 21865, 5416,
    14686, 31510, 27552, 18714, 14405, 4381, 23780, 22884, 22461, 21636, 14555,
    18915, 10811, 19134, 3344, 13642, 21645, 16896, 22927, 6431, 29065, 1824,
    14972, 8963, 13984, 26053, 22416, 11271, 28697, 17604, 18051, 5015, 15407,
    6465
]

with open('./text_spatial_rel_phrases.json', 'r') as f:
    text_data = json.load(f)


# %%
# HELPER functions
def image_grid(imgs, rows, cols):
    assert len(imgs) == rows * cols

    w, h = imgs[0].size
    grid = Image.new('RGB', size=(cols * w, rows * h))
    grid_w, grid_h = grid.size

    for i, img in enumerate(imgs):
        grid.paste(img, box=(i % cols * w, i // cols * h))
    return grid


# HELPERS
def increment_dict(d, k, v, inc_type="list"):
    inc = [v] if inc_type == "list" else v
    if k in d:
        d[k] = d[k] + inc
    else:
        d[k] = inc
    return d


def compute_recall(obj1, obj2, detected, N):
    if obj1 in detected and obj2 in detected:
        count = 2
    elif obj1 in detected or obj2 in detected:
        count = 1
    else:
        count = 0

    return count / N, count


# %%
# OBJECT DETECTION
def process_detection(outs, obj1, obj2, rel):
    objects = [obj1, obj2]
    boxes, scores, labels = outs[0]["boxes"], outs[0]["scores"], outs[0]["labels"]

    det_bbox, det_scores, det_labels = [], [], []
    for box, score, label in zip(boxes, scores, labels):
        if score > 0.1:
            det_bbox.append(box.tolist())
            det_scores.append(score.tolist())
            det_labels.append(objects[label.item()])

    det_centroid = [((box[0] + box[2]) / 2, (box[1] + box[3]) / 2) for box in det_bbox]
    N = len(det_centroid)

    if obj1 in det_labels and obj2 in det_labels:
        recall = 2
    elif obj1 in det_labels:
        recall = 1
    elif obj2 in det_labels:
        recall = 1
    else:
        recall = 0

    sra = 0
    if obj1 in det_labels and obj2 in det_labels:
        idx1 = np.where(np.array(det_labels) == obj1)[0]
        idx2 = np.where(np.array(det_labels) == obj2)[0]

        # atleast one of the bbox pairs should follow the relationship
        for i1, i2 in itertools.product(idx1.tolist(), idx2.tolist()):
            xdist = det_centroid[i1][0] - det_centroid[i2][0]
            ydist = det_centroid[i1][1] - det_centroid[i2][1]
            if rel == "to the left of" and xdist < 0:
                sra = 1
                break
            if rel == "to the right of" and xdist > 0:
                sra = 1
                break
            if rel == "above" and ydist < 0:
                sra = 1
                break
            if rel == "below" and ydist > 0:
                sra = 1
                break
    return {
        "classes": det_labels, "centroid": det_centroid, "recall": recall, "sra": sra, "rel_type": rel
    }


# %%
# VISOR
def get_visor_beifen(results, obj1, obj2, rel, uniq_id):
    N_D, N_R = {}, {}
    objacc_both, objacc_A, objacc_B = 0, 0, 0
    avg_visor, avg_sra = 0, 0
    both_count, count = 0, 0
    visor_1, visor_2, visor_3, visor_4 = {}, {}, {}, {}
    visor_by_uniq_id = {}

    for img_id, rr in results.items():

        detected = rr["classes"]
        N_R = increment_dict(N_R, rr["recall"], 1)
        N_D = increment_dict(N_D, len(detected), 1)

        recall = rr["recall"] / 2
        both = int(obj1 in detected and obj2 in detected)
        sra = rr["sra"]
        objacc_both = objacc_both + both
        objacc_A = objacc_A + int(obj1 in detected)
        objacc_B = objacc_B + int(obj2 in detected)

        avg_sra = avg_sra + sra
        avg_visor = avg_visor + both * sra
        both_count = both_count + both
        count = count + 1

        if both == 1:
            visor_by_uniq_id = increment_dict(
                visor_by_uniq_id, uniq_id, both * sra)

    # visor scores
    visor_uncond = 100 * avg_sra / (count + 1e-6)
    visor_cond = 100 * avg_visor / (both_count + 1e-6)
    visor_per_text = get_visor_per_text(visor_by_uniq_id)

    # objacc scores
    objacc = [100 * objacc_A / count, 100 * objacc_B / count, 100 * objacc_both / count]

    return visor_uncond, visor_cond, visor_per_text, objacc

def get_visor(results, obj1, obj2, rel, uniq_id):
    N_D, N_R = {}, {}
    objacc_both, objacc_A, objacc_B = 0, 0, 0
    avg_visor, avg_sra = 0, 0
    both_count, count = 0, 0
    visor_1, visor_2, visor_3, visor_4 = {}, {}, {}, {}
    visor_by_uniq_id = {}

    for img_id, rr in results.items():

        detected = rr["classes"]
        N_R = increment_dict(N_R, rr["recall"], 1)
        N_D = increment_dict(N_D, len(detected), 1)

        recall = rr["recall"] / 2
        both = int(obj1 in detected and obj2 in detected)
        sra = rr["sra"]
        objacc_both = objacc_both + both
        count = count + 1


    # objacc scores
    objacc = 100 * objacc_both / count

    return objacc


def get_visor_per_text(visor_by_uniq_id):
    visor_1, visor_2, visor_3, visor_4 = 0, 0, 0, 0
    for uniq_id, scores in visor_by_uniq_id.items():
        if sum(scores) >= 4:
            visor_4 = visor_4 + 1
        if sum(scores) >= 3:
            visor_3 = visor_3 + 1
        if sum(scores) >= 2:
            visor_2 = visor_2 + 1
        if sum(scores) >= 1:
            visor_1 = visor_1 + 1

    NUM_UNIQ = len(visor_by_uniq_id) + 1e-6

    return [100 * visor_1 / NUM_UNIQ, 100 * visor_2 / NUM_UNIQ, 100 * visor_3 / NUM_UNIQ, 100 * visor_4 / NUM_UNIQ]

pp = 0
objacc_sum = 0
# %%
for uniq_id in range(31680):
    pp += 1
    # uniq_id = random.choice(viz_ids)
    free_form_prompt = text_data[uniq_id]["text"]
    obj1 = text_data[uniq_id]["obj_1_attributes"][0]
    obj2 = text_data[uniq_id]["obj_2_attributes"][0]
    rel = text_data[uniq_id]["rel_type"]

    # print("UNIQ_ID: {}; \tTEXT: {}; \tOBJ-A: {}; \tOBJ-B: {}; \tREL: {}".format(uniq_id, free_form_prompt, obj1, obj2, rel))
    # %%
    # LOAD ALL 4 IMAGES FOR ALL MODELS AND DISPLAY THEM
    all_images = {}
    for mo in MODELS:
        images = []
        for i in range(1):
            img_id = "{}_{}".format(uniq_id, i)
            impath = URL_PREFIX + "/{}.png".format(img_id)
            im = Image.open(impath)
            images.append(im)
        all_images[mo] = images
    # %%
    # process object detection
    for mo in MODELS:
        results = {}
        images = all_images[mo]
        texts = [["a photo of a {}".format(obj1), "a photo of a {}".format(obj2)]]
        grid = image_grid(images, 1, 1)
        # display(grid)
        for i in range(1):
            image = images[i]
            img_id = "{}_{}".format(uniq_id, i)
            with torch.no_grad():
                inputs = processor(text=texts, images=image, return_tensors="pt").to(device)
                outputs = model(**inputs)
                target_sizes = torch.Tensor([image.size[::-1]]).to(device)
                outs = processor.post_process(outputs=outputs, target_sizes=target_sizes)
            results[img_id] = process_detection(outs, obj1, obj2, rel)
        objacc = get_visor(results, obj1, obj2, rel, uniq_id)
        objacc_sum += objacc
        objacc_ave = objacc_sum / pp
        print("Samples: {:.2f}, Object Accuracy: {:.2f}, Object Accuracy (ave): {:.2f}".format(pp, objacc, objacc_ave))
