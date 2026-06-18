import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms
from copy import deepcopy

import math
import torch.nn.functional as F
import pandas as pd

import torch
import torch.nn as nn
import matplotlib.pyplot as plt


import torch
import torch.nn as nn
import torch.nn.functional as F
import itertools

class TokenConflictLoss(nn.Module):
    def __init__(self, num_nouns, tau=0.3):
        """
        num_nouns: int, number of noun-level concepts
        tau: float, threshold used in inter-object conflict loss
        """
        super(TokenConflictLoss, self).__init__()
        self.num_nouns = num_nouns
        self.tau = tau
        # Learnable gamma for each noun concept
        self.gamma = nn.Parameter(torch.ones(num_nouns))

    def compute_belief(self, attn_map, seg_mask, gamma):
        """
        attn_map: (B, H, W)
        seg_mask: (B, H, W)
        gamma: scalar tensor
        """
        return gamma * attn_map * seg_mask

    def compute_conflict(self, b_v, b_w, m_v, m_w):
        """
        Compute conflict coefficient K^{v,w} and overlapping region
        b_v, b_w: (B, H, W)
        m_v, m_w: (B, H, W)
        """
        overlap = (m_v > 0) & (m_w > 0)
        K_vw = torch.zeros_like(b_v)
        K_vw[overlap] = b_v[overlap] * b_w[overlap]
        return K_vw, overlap

    def forward(self, attn_maps, seg_masks):
        """
        attn_maps: list of tensors [(B, H, W), ...], length = N (num_nouns)
        seg_masks: list of tensors [(B, H, W), ...], length = N
        Returns: L_intra, L_inter
        """
        assert len(attn_maps) == len(seg_masks) == self.num_nouns

        beliefs = []
        for n in range(self.num_nouns):
            b_n = self.compute_belief(attn_maps[n], seg_masks[n], self.gamma[n])
            beliefs.append(b_n)

        # Compute intra-object consistency loss
        L_intra = sum([(1 - b_n) ** 2 for b_n in beliefs])
        L_intra = torch.stack([b.mean() for b in L_intra]).sum()

        # Compute inter-object conflict loss
        L_inter = 0.0
        count = 0
        for v, w in itertools.combinations(range(self.num_nouns), 2):
            K_vw, overlap = self.compute_conflict(
                beliefs[v], beliefs[w], seg_masks[v], seg_masks[w]
            )
            K_overlap = K_vw[overlap]
            if K_overlap.numel() == 0:
                continue
            denom = 1.0 - K_overlap + 1e-6
            conflict_term = (K_overlap / denom) * torch.sigmoid(K_overlap - self.tau)
            L_inter += conflict_term.mean()
            count += 1

        if count > 0:
            L_inter /= count

        return L_inter



def aggregate_evidence(evidence_list, beta_token):
    """
    输入: 多个 (B, 1, X, X) 的证据张量，列表
    输出: 聚合后的 belief 和 uncertainty
    """
    eps = 1e-8
    num_sources = len(evidence_list)

    # 将每个证据 e 转成 belief 和 uncertainty
    beliefs = [e / (e + 1 + eps) for e in evidence_list]
    uncertainties = [1 / (e + 1 + eps) for e in evidence_list]

    # 初始化
    combined_belief = beliefs[0]
    combined_uncertainty = uncertainties[0]

    for i in range(1, num_sources):
        b1 = combined_belief
        u1 = combined_uncertainty
        b2 = beliefs[i]
        u2 = uncertainties[i]

        # 冲突项
        conflict = (b1 * (1 - b2)) + ((1 - b1) * b2)
        conflict = torch.clamp(conflict, min=eps, max=1.0)  # 防止除以0

        # Dempster combination
        combined_belief = (b1 * b2) / conflict
        combined_uncertainty = (u1 * u2) / conflict

    return combined_belief, combined_uncertainty


def TE_evidential_loss(evidence_list, beta_token):
    # 聚合所有证据
    evidence_list = [torch.as_tensor(e, dtype=torch.float32) for e in evidence_list]
    combined_belief, combined_uncertainty = aggregate_evidence(evidence_list, beta_token)

    # 计算 loss
    # 目标是combined_belief尽量大（集中注意力到目标上）
    loss = torch.mean((1 - combined_belief) ** 2)

    return loss


def my_fuse_opinions(alpha1, alpha2, combine_method, beta):
    num_classes = 2
    strength = [alpha1.sum(dim=1, keepdim=True), alpha2.sum(dim=1, keepdim=True)]
    evidence = [a - beta for a in [alpha1, alpha2]]
    belief = [e / s.expand_as(e) for e, s in zip(evidence, strength)]
    uncertainty = [num_classes / s for s in strength]

    if combine_method == 'constraint':
        # Calculate conflict measure
        belief_product = torch.bmm(
            belief[0].unsqueeze(2),
            belief[1].unsqueeze(1)
        ).sum(dim=(1, 2)) - (belief[0] * belief[1]).sum(dim=1)
        conflict = 1 - belief_product.unsqueeze(1)
        fused_belief = (belief[0] * belief[1] + belief[0] * uncertainty[1] + belief[1] * uncertainty[0]) / conflict
        fused_uncertainty = (uncertainty[0] * uncertainty[1]) / conflict

    elif combine_method == 'cumulative':
        combined_uncertainty = uncertainty[0] + uncertainty[1] - uncertainty[0] * uncertainty[1]
        fused_belief = (belief[0] * uncertainty[1] + belief[1] * uncertainty[0]) / combined_uncertainty
        fused_uncertainty = (uncertainty[0] * uncertainty[1]) / combined_uncertainty
    else:
        raise ValueError(f"Invalid combination method: {combine_method}")

    fused_strength = num_classes / fused_uncertainty

    return fused_belief * fused_strength + beta

def my_combine_views(alphas, beta):
    """Combine multiple views according to specified strategy"""
    # Validate combination method
    COMBINE_STRATEGIES = {
        'hybrid': ('cumulative', 'constraint'),
        'BCF': ('constraint', 'constraint'),
        'CBF': ('cumulative', 'cumulative')
    }

    local_method, global_method = COMBINE_STRATEGIES['hybrid']

    # Combine local views
    combined_alpha = alphas[0]
    for i in range(1, len(alphas)):
        combined_alpha = my_fuse_opinions(
            combined_alpha,
            alphas[i],
            local_method,
            beta
        )

    return combined_alpha

def my_compute_alpha(evidence, beta):
    alpha = dict()
    for evi in range(len(evidence)):
        alpha[evi] = evidence[evi] + beta
    # alpha['combined'] = my_combine_views(alpha, beta)
    return alpha

def my_fuse_opinions_token(alpha1, alpha2, combine_method, beta):
    num_classes = 2
    strength = [alpha1.sum(dim=1, keepdim=True), alpha2.sum(dim=1, keepdim=True)]
    evidence = [a - beta for a in [alpha1, alpha2]]
    belief = [e / s.expand_as(e) for e, s in zip(evidence, strength)]
    uncertainty = [num_classes / s for s in strength]

    if combine_method == 'constraint':
        # Calculate conflict measure
        belief_product = torch.bmm(
            belief[0].unsqueeze(2),
            belief[1].unsqueeze(1)
        ).sum(dim=(1, 2)) - (belief[0] * belief[1]).sum(dim=1)
        conflict = 1 - belief_product.unsqueeze(1)
        fused_belief = (belief[0] * belief[1] + belief[0] * uncertainty[1] + belief[1] * uncertainty[0]) / conflict
        fused_uncertainty = (uncertainty[0] * uncertainty[1]) / conflict

    elif combine_method == 'cumulative':
        combined_uncertainty = uncertainty[0] + uncertainty[1] - uncertainty[0] * uncertainty[1]
        fused_belief = (belief[0] * uncertainty[1] + belief[1] * uncertainty[0]) / combined_uncertainty
        fused_uncertainty = (uncertainty[0] * uncertainty[1]) / combined_uncertainty
    else:
        raise ValueError(f"Invalid combination method: {combine_method}")

    fused_strength = num_classes / fused_uncertainty

    return fused_belief, conflict

def my_combine_views_token(alphas, beta):
    """Combine multiple views according to specified strategy"""
    # Validate combination method
    COMBINE_STRATEGIES = {
        'hybrid': ('cumulative', 'constraint'),
        'BCF': ('constraint', 'constraint'),
        'CBF': ('cumulative', 'cumulative')
    }

    local_method, global_method = COMBINE_STRATEGIES['BCF']

    # Combine local views
    combined_alpha = alphas[0]
    if len(alphas) > 1:
        for i in range(1, len(alphas)):
            fused_belief, fused_uncertainty = my_fuse_opinions_token(
                combined_alpha,
                alphas[i],
                local_method,
                beta
            )
    else:
        fused_belief = torch.tensor([1.0]).float()
        fused_uncertainty = torch.tensor([0.0]).float()

    return fused_belief, fused_uncertainty

def my_compute_alpha_token(evidence, beta):
    alpha = dict()
    for evi in range(len(evidence)):
        alpha[evi] = evidence[evi] + beta
    fused_belief, fused_uncertainty = my_combine_views_token(alpha, beta)
    return fused_belief, fused_uncertainty


def my_calculate_loss(alpha, label, beta, global_step):
    annealing_coef = min(1, global_step / 100)
    loss = {"ce": 0.0, "kl": 0.0, "total": 0.0}

    for key, a in alpha.items():
        ce, kl = my_compute_individual_loss(label, a, beta)
        loss["ce"] += ce
        loss["kl"] += kl
        loss["total"] += ce + annealing_coef * kl

    return {k: torch.mean(v) for k, v in loss.items()}

def my_compute_individual_loss(label, alpha, beta):
    beta = beta.expand(alpha.shape[0], -1)
    # Cross entropy calculation
    S = alpha.sum(dim=1, keepdim=True)
    ce = label * (torch.digamma(S) - torch.digamma(alpha)).sum(dim=1, keepdim=True)

    # KL divergence calculation
    alp = label * beta + (1 - label) * alpha
    kl = my_kl_divergence(alp, beta)

    return ce, kl

def my_kl_divergence(alpha, beta):
    # NOTE: beta = alpha - E = base_rate x classes

    S_alpha = torch.sum(alpha, dim=1, keepdim=True)
    S_beta = torch.sum(beta, dim=1, keepdim=True)
    lnB = torch.lgamma(S_alpha) - torch.sum(torch.lgamma(alpha), dim=1, keepdim=True)
    lnB_uni = torch.sum(torch.lgamma(beta), dim=1, keepdim=True) - torch.lgamma(S_beta)
    dg0 = torch.digamma(S_alpha)
    dg1 = torch.digamma(alpha)
    kl = torch.sum((alpha - beta) * (dg1 - dg0), dim=1, keepdim=True) + lnB + lnB_uni
    return kl

def to_torch(ndarray):
    if isinstance(ndarray, (int, float)):
        ndarray = torch.tensor([ndarray])
    elif isinstance(ndarray, (tuple, list)):
        ndarray = torch.tensor(ndarray)
    elif isinstance(ndarray, np.ndarray):
        ndarray = torch.from_numpy(ndarray)
    return ndarray

device = torch.device('cuda:0')
beta = to_torch([1, 1]).to(device)

# def kl_divergence(alpha, num_classes):
#     ones = torch.ones([1, num_classes], dtype=torch.float32)
#     sum_alpha = torch.sum(alpha, dim=1, keepdim=True)
#     first_term = (
#         torch.lgamma(sum_alpha)
#         - torch.lgamma(alpha).sum(dim=1, keepdim=True)
#         + torch.lgamma(ones).sum(dim=1, keepdim=True)
#         - torch.lgamma(ones.sum(dim=1, keepdim=True))
#     )
#     second_term = (
#         (alpha - ones)
#         .mul(torch.digamma(alpha) - torch.digamma(sum_alpha))
#         .sum(dim=1, keepdim=True)
#     )
#     kl = first_term + second_term
#     return kl
#
#
# def edl_loss(func, y, alpha, epoch_num, num_classes, annealing_step, useKL=True):
#     S = torch.sum(alpha, dim=1, keepdim=True)  # S为迪利克雷分布的长度
#
#     A = torch.sum(y * (func(S) - func(alpha)), dim=1, keepdim=True)
#
#     if not useKL:
#         return A
#
#     annealing_coef = torch.min(
#         torch.tensor(1.0, dtype=torch.float32),
#         torch.tensor(epoch_num / annealing_step, dtype=torch.float32),
#     )
#
#     kl_alpha = (alpha - 1) * (1 - y) + 1
#     kl_div = annealing_coef * kl_divergence(kl_alpha, num_classes)
#     return A + kl_div
#
# def edl_digamma_loss(alpha, target, epoch_num, num_classes, annealing_step):
#     loss = edl_loss(torch.digamma, target, alpha, epoch_num, num_classes, annealing_step)
#     return torch.mean(loss)
#
# def get_loss(evidences, target, epoch_num, num_classes, annealing_step, device):  # annealing_step=50
#     loss_acc = 0.0
#     for v in range(len(evidences)):
#         alpha = evidences[v] + 1  # alpha为迪利克雷分布的参数
#         loss_acc += edl_digamma_loss(alpha, target, epoch_num, num_classes, annealing_step)
#     loss_acc = loss_acc / len(evidences)
#     loss = loss_acc
#     return loss

SD14_TO_SD21_RATIO = 1.5

# get token index in text
def get_word_idx(text: str, tgt_word, tokenizer):

    tgt_word = tgt_word.lower()

    # ignore the first and last token
    encoded_text = tokenizer.encode(text)[1:-1]
    encoded_tgt_word = tokenizer.encode(tgt_word)[1:-1]

    # find the idx of target word in text
    first_token_idx = -1
    for i in range(len(encoded_text)):
        if encoded_text[i] == encoded_tgt_word[0]:

            if len(encoded_text) > 0:
                # check the following 
                following_match = True
                for j in range(1, len(encoded_tgt_word)):
                    if encoded_text[i + j] != encoded_tgt_word[j]:
                        following_match = False
                if not following_match:
                    continue
            # for a single encoded idx, just take it
            first_token_idx = i

            break

    assert first_token_idx != -1, "word not in text"

    # add 1 for sot token
    tgt_word_tokens_idx_ls = [i + 1 + first_token_idx for i in range(len(encoded_tgt_word))]

    # sanity check
    encoded_text = tokenizer.encode(text)

    decoded_token_ls = []

    for word_idx in tgt_word_tokens_idx_ls:
        text_decode = tokenizer.decode([encoded_text[word_idx]]).strip("#")
        decoded_token_ls.append(text_decode)

    decoded_tgt_word = "".join(decoded_token_ls)
    
    tgt_word_ls = tgt_word.split(" ")
    striped_tgt_word = "".join(tgt_word_ls).strip("#")

    assert decoded_tgt_word == striped_tgt_word, "decode_text != striped_tar_wd"

    return tgt_word_tokens_idx_ls

# get attn loss by resolution
  # _gt_seg_list = list[[1, 1, 512, 512], [1, 1, 512, 512], [1, 1, 512, 512]]
  # word_token_idx_ls = [[5, 6], [14, 15], [2, 3]]
  # res = 8
  # input_attn_map_ls = list[[8, 8, 8, 77]]
  # is_training_sd21 = False
def get_grounding_loss_by_layer(_gt_seg_list, word_token_idx_ls, res,
                                input_attn_map_ls, is_training_sd21, epoch, PE_Net, TE_Net, global_step, step):
    if is_training_sd21:
        # training with sd21, using resolution 768 = 512 * 1.5
        res = int(SD14_TO_SD21_RATIO * res)

    gt_seg_list = deepcopy(_gt_seg_list)  # list[[1, 1, 512, 512], [1, 1, 512, 512], [1, 1, 512, 512]]

    # reszie gt seg map to the same size with attn map
    resize_transform = transforms.Resize((res, res))  # (8, 8)

    for i in range(len(gt_seg_list)):  # 3
        gt_seg_list[i] = resize_transform(gt_seg_list[i])
        gt_seg_list[i] = gt_seg_list[i].squeeze(0) # 1, 1, res, res => 1, res, res
        # add binary
        gt_seg_list[i] = (gt_seg_list[i] > 0.0).float()  # gt_seg_list[i] = [1, 8, 8], values=0, 1
    # gt_seg_list = list[[1, 8, 8], [1, 8, 8], [1, 8, 8]]


    ################### token loss start ###################
    # Following code is adapted from
    # https://github.com/silent-chen/layout-guidance/blob/08b687470f911c7f57937012bdf55194836d693e/utils.py#L27
    token_loss = 0.0
    token_loss_evidence_final = 0.0
    token_loss_evidence_sum = 0.0
    pp = 0
    token_inter = 0
    for attn_map in input_attn_map_ls:  # input_attn_map_ls = list[[8, 8, 8, 77]]
        pp +=1
        evidences_token = []
        i_token = 0

        b, H, W, j = attn_map.shape  # [8, 8, 8, 77]


        for i in range(len(word_token_idx_ls)):  # 3 [[word1 token_idx1, word1 token_idx2, ...], [word2 token_idx1, word2 token_idx2, ...]]
            obj_loss = 0.0

            single_word_idx_ls = word_token_idx_ls[i]  # [token_idx1, token_idx2, ...]
            mask = gt_seg_list[i]  # [1, 8, 8]

            obj_position_intra_noun = 0
            for obj_position in single_word_idx_ls:  # [5, 6]
                # ca map obj shape 8 * 16 * 16

                ca_map_obj = attn_map[:, :, :, obj_position].reshape(b, H, W)  # [8, 8, 8]

                # ---------------------------------------
                obj_position_intra_noun +=  ca_map_obj
                # ---------------------------------------

                # intra loss
                activation_value_ori = (ca_map_obj * mask).reshape(b, -1).sum(dim=-1)/ca_map_obj.reshape(b, -1).sum(dim=-1)  # [8]
                obj_loss += (1.0 - torch.mean(activation_value_ori)) ** 2
            token_loss += (obj_loss/len(single_word_idx_ls))

            # ---------------------------------------
            obj_position_intra_noun = torch.mean(obj_position_intra_noun, dim=0, keepdim=True)
            evidences_token.append(obj_position_intra_noun)
            # ---------------------------------------

        # ---------------------------------------
        evidences_token = torch.stack(evidences_token)
        loss_fn = TokenConflictLoss(num_nouns=len(evidences_token), tau=0.3)
        token_inter += loss_fn(evidences_token, gt_seg_list)
    token_inter = token_inter / len(input_attn_map_ls)
        # ---------------------------------------


    # normalize with len words
    token_loss_final = token_loss / len(word_token_idx_ls) + token_inter
    ################## token loss end ##########################

    ################## pixel loss start ######################
    # average cross attention map on different layers
    avg_attn_map_ls = []
    for i in range(len(input_attn_map_ls)):  # input_attn_map_ls = list[[8, 8, 8, 77]]
        avg_attn_map_ls.append(
            input_attn_map_ls[i].reshape(-1, res, res, input_attn_map_ls[i].shape[-1]).mean(0)  # list[[8, 8, 77]]
        )

    avg_attn_map = torch.stack(avg_attn_map_ls, dim=0)  # [L, 8, 8, 77]
    avg_attn_map = avg_attn_map.sum(0) / avg_attn_map.shape[0]
    avg_attn_map = avg_attn_map.unsqueeze(0)  # [1, 8, 8, 77]

    bce_loss_func = nn.BCELoss()
    pixel_loss = 0.0
    qq = 0
    for i in range(len(word_token_idx_ls)):  # word_token_idx_ls = [[5, 6], [14, 15], [2, 3]]
        qq += 1
        word_cross_attn_ls = []
        evidences_pixels = {}
        for token_idx in word_token_idx_ls[i]:
            word_cross_attn_ls.append(
                avg_attn_map[..., token_idx]
            )
        word_cross_attn_ls = torch.stack(word_cross_attn_ls, dim=0).sum(dim=0)  # [1, 8, 8]

        #--------------- pixel evidence loss -------------------------------------------------------------------------------------------------------#
        # extract word_cross_attn_ls_pixels evidences
        word_cross_attn_ls_pixels = word_cross_attn_ls.view(-1)
        word_cross_attn_ls_pixels_0 = 1 - word_cross_attn_ls_pixels
        word_cross_attn_ls_pixels_1 = word_cross_attn_ls_pixels
        word_cross_attn_ls_pixels_prob_matrix = torch.stack([word_cross_attn_ls_pixels_0, word_cross_attn_ls_pixels_1], dim=1)  # [n, 2]
        word_cross_attn_ls_pixels_prob_matrix_ = PE_Net(word_cross_attn_ls_pixels_prob_matrix)

        evidences_pixels[0] = word_cross_attn_ls_pixels_prob_matrix_

        # transfer mask to one_hot
        word_cross_attn_mask = gt_seg_list[i].view(-1).to(torch.int64)
        word_cross_attn_mask = F.one_hot(word_cross_attn_mask, 2)

        # compute evidence loss
        alpha = my_compute_alpha(evidences_pixels, beta)
        loss_evi = my_calculate_loss(alpha, word_cross_attn_mask, beta, global_step=epoch)
        loss_evi_pixel = 0.01*loss_evi["total"] + bce_loss_func(word_cross_attn_ls, gt_seg_list[i])  # gt_seg_list[i]=[1, 8, 8]
        pixel_loss += loss_evi_pixel
        #---------------pixel evidence loss -------------------------------------------------------------------------------------------------------#


    # average with len word_token_idx_ls
    pixel_loss = pixel_loss / len(word_token_idx_ls)
    ################## pixel loss end #########################

    return {
        "token_loss" : token_loss_final,
        "pixel_loss": pixel_loss,
    }

