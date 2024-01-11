import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import einops
from fancy_einsum import einsum
import random
from pathlib import Path
import plotly.express as px
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from functools import partial
import circuitsvis as cv

import transformer_lens
import transformer_lens.utils as utils
from transformer_lens.hook_points import (
    HookedRootModule,
    HookPoint,
)  # Hooking utilities
from transformer_lens import HookedTransformer, HookedTransformerConfig, FactoredMatrix, ActivationCache
from numpy.linalg import norm
from pprint import pprint

import argparse
import json
# Create an ArgumentParser object
parser = argparse.ArgumentParser()
#
## Define command-line arguments
parser.add_argument('--model', help='word to search')
parser.add_argument('--head', default = 'memory', help='word to search')
from pprint import pprint
#parser.add_argument('word', help='word to search')
#
args = parser.parse_args()
head = args.head

torch.set_grad_enabled(False)
device = "cuda" if torch.cuda.is_available() else "cpu"
model_name = args.model
model = HookedTransformer.from_pretrained(model_name, device = device)

def get_tuning_data_gt(path = f'{model_name}_abs_icl_country_all_decoded.json'):
    with open(path, 'r') as f:
        all_rank = json.load(f)
    for k, v in all_rank.items():
        if len(v)< 20:
            end = len(v)
        else:
            end = 20
        sampled= []
        while len(sampled) < end:
            pick = v[random.choice(range(len(v)))]
            if "Vatican City" in pick['inputs']:
                continue
            if pick['gt_correct'] == 1:
                sampled.append(pick)
        all_rank[k] = sampled
    with open(f'{model_name}_tuning_set_gt.json', 'w') as f:
        json.dump(all_rank, f)


def get_tuning_data_ct(path = f'{model_name}_abs_icl_country_all_decoded.json'):
    with open(path, 'r') as f:
        all_rank = json.load(f)

    for k, v in all_rank.items():
        if len(v)< 20:
            end = len(v)
        else:
            end = 20
        sampled= []
        while len(sampled) < end:
            pick = v[random.choice(range(len(v)))]
            if "Vatican City" in pick['inputs']:
                continue
            if pick['correct'] == 1:
                sampled.append(pick)
        all_rank[k] = sampled
    with open(f'{model_name}_tuning_set_ct.json', 'w') as f:
        json.dump(all_rank, f)

def change(i):
    with open(f'{model_name}_abs_icl_country_memory_neg.json', 'r') as f:
        all_rank = json.load(f)

    with open(f'{model_name}_tuning_set_ct.json', 'r') as f:
       chang = json.load(f)

    th = all_rank[str(i)]
        
    end = 20
    sampled= []
    while len(sampled) < end:
        pick = th[random.choice(range(len(th)))]
        if "Vatican City" in pick['inputs']:
            continue
        if "South Korea is Minsk" in pick['inputs']:
            continue
        if pick['correct'] == 1:
            sampled.append(pick)
    chang[str(i)] = sampled
    with open(f'{model_name}_tuning_set_ct.json', 'w') as f:
        json.dump(chang, f)


def head_upweight_hook(
    value,#: Float[torch.Tensor, "batch pos head_index d_head"],
    hook: HookPoint,
    head_idx,
    scale
):# -> Float[torch.Tensor, "batch pos head_index d_head"]:
    value[:, :, head_idx, :] *=scale
    return value

def get_data(sample_gt, rank, batch_size):
    
    data = sample_gt[rank]
    batch_data = []
    for i in range(0, len(data), batch_size):
        batch = data[i:i+batch_size]
        batch_data.append(batch)

    prompts= [sample['inputs'] for sample in sample_gt[rank]]
    batch_prompts = []
    for i in range(0, len(prompts), batch_size):
        batch = prompts[i:i+batch_size]
        batch_prompts.append(batch)

    
    ct_answers = [sample['targets'][0] for sample in sample_gt[rank]]
    batch_ct_answers = []
    for i in range(0, len(ct_answers), batch_size):
        batch = ct_answers[i:i+batch_size]
        batch_ct_answers.append(batch)


    gt_answers = [sample['targets'][1] for sample in sample_gt[rank]]
    batch_gt_answers = []
    for i in range(0, len(gt_answers), batch_size):
        batch = gt_answers[i:i+batch_size]
        batch_gt_answers.append(batch)
    

    answer_tokens = torch.stack([torch.stack([model.to_tokens(sample['targets'][0])[0][1],
                model.to_tokens(sample['targets'][1])[0][1]]) for sample in sample_gt[rank]])
    batch_answer_tokens = []
    for i in range(0, len(answer_tokens), batch_size):
        batch = answer_tokens[i:i+batch_size]
        batch_answer_tokens.append(batch)


    batch_answer_residual_directions = []
    batch_logit_diff_directions = []
    for i in batch_answer_tokens:
        answer_residual_directions = model.tokens_to_residual_directions(i)
        batch_answer_residual_directions.append(answer_residual_directions)
        logit_diff_directions = answer_residual_directions[:, 0] - answer_residual_directions[:, 1]
        batch_logit_diff_directions.append(logit_diff_directions)
    

    tokens =  [model.to_tokens(i, prepend_bos = False).cuda() for i in prompts]
    batch_tokens = []
    for i in range(0, len(tokens), batch_size):
        batch = tokens[i:i+batch_size]
        batch_tokens.append(batch)
    
    return batch_data, batch_prompts, batch_tokens, batch_ct_answers, batch_gt_answers, batch_logit_diff_directions


def generate_weighted_output(memory, token, scale):
    model.reset_hooks()
    if model_name == 'pythia-1.4b':
        if head == 'memory':
            downweight_hook = partial(head_upweight_hook, head_idx=7, scale= scale) 
            model.blocks[15].attn.hook_v.add_hook(downweight_hook)
        else:
            downweight_hook = partial(head_upweight_hook, head_idx=14, scale= scale) 
            model.blocks[19].attn.hook_v.add_hook(downweight_hook)

    if model_name == 'pythia-2.8b':
        if head =='memory':
            downweight_hook = partial(head_upweight_hook, head_idx=17, scale= scale) ## memory head
            model.blocks[17].attn.hook_v.add_hook(downweight_hook) ## scale up (positive)-> gt ## scale down (negative) -> icl
        else:
            downweight_hook = partial(head_upweight_hook, head_idx=31, scale= scale) ## icl head
            model.blocks[17].attn.hook_v.add_hook(downweight_hook) ## scale up (positive) -> icl ## scale down (negative) -> gt

    if model_name == 'gpt2-xl':
        if head =='memory':
            downweight_hook = partial(head_upweight_hook, head_idx=19, scale= scale) ## memory head
            model.blocks[35].attn.hook_v.add_hook(downweight_hook) ## scale up (positive)-> gt ## scale down (negative) -> icl
        else:
            downweight_hook = partial(head_upweight_hook, head_idx=18, scale= scale) ## icl head
            model.blocks[41].attn.hook_v.add_hook(downweight_hook) ## scale up (positive) -> icl ## scale down (negative) -> gt

    return model.tokenizer.decode(model.generate(token,  max_new_tokens = len(token)+15, temperature =0)[0])

def intervention(head, tokens, ct_answers, gt_answers, scale):
    ct = []
    gt = []
    for idx, d in enumerate(tokens):

        a = generate_weighted_output(head, d,  scale)
        pprint(a)
        model.reset_hooks()
        import re
        answer = a[a.index("A:"): ]
        if "Q:" in answer:
            answer = answer[:answer.index("Q:")]
        if len(re.findall(ct_answers[idx],answer))> 0:      
            ct.append(1)
        else:
            ct.append(0) 

        if len(re.findall(gt_answers[idx],answer))>0:         
            gt.append(1)
        else:
            gt.append(0)     

    return np.sum(ct), np.sum(gt)

    
def tune_curve_freq(tokens, ct_answers, gt_answers):
    result_ct = []
    result_gt = []
    scale = []
    if head == 'memory':
        r = np.arange(-5, 6, 0.1)
    else:
        r = np.arange(-10.0, 11.0, 0.5)

    for i in r:
        scale.append(i)
        ct_acc, gt_acc = intervention(head, tokens, ct_answers, gt_answers, i)
        result_ct.append(ct_acc)
        result_gt.append(gt_acc)

    return scale, result_ct, result_gt

def run_all(batch_size):
    all_tuned_curve = {}
    with open(f'{model_name}_tuning_set_ct.json', 'r') as f:
        all_rank = json.load(f)
    for r in range(10):
        result_all_ct = []
        result_all_gt = []
        batch_data, batch_prompts, batch_tokens, batch_ct_answers, batch_gt_answers, batch_logit_diff_directions = get_data(all_rank, str(r), batch_size)
        for idx, i in enumerate(batch_prompts):
            data, prompts, tokens, ct_answers, gt_answers, logit_diff_directions = batch_data[idx], batch_prompts[idx], batch_tokens[idx], batch_ct_answers[idx], batch_gt_answers[idx], batch_logit_diff_directions[idx]
            scale, result_ct, result_gt =  tune_curve_freq(tokens, ct_answers, gt_answers)
            result_all_ct.append(result_ct)
            result_all_gt.append(result_gt)

        result_avg_ct = np.sum(result_all_ct, axis = 0)/20
        result_avg_gt = np.sum(result_all_gt, axis = 0)/20
        all_tuned_curve[r] = {'scale': scale, 'result_ct': result_avg_ct.tolist(), 'result_gt': result_avg_gt.tolist()}

        print(all_tuned_curve, flush = True)

    with open(f'{model_name}_{head}_tuned_curve_ct_m_camera_ready.json', 'w') as f:
        json.dump(all_tuned_curve, f)
    
    
    all_tuned_curve = {}
    with open(f'{model_name}_tuning_set_gt.json', 'r') as f:
        all_rank = json.load(f)
    for r in range(10):
        result_all_ct = []
        result_all_gt = []
        batch_data, batch_prompts, batch_tokens, batch_ct_answers, batch_gt_answers, batch_logit_diff_directions = get_data(all_rank, str(r), batch_size)
        for idx, i in enumerate(batch_prompts):
            data, prompts, tokens, ct_answers, gt_answers, logit_diff_directions = batch_data[idx], batch_prompts[idx], batch_tokens[idx], batch_ct_answers[idx], batch_gt_answers[idx], batch_logit_diff_directions[idx]
            scale, result_ct, result_gt =  tune_curve_freq(tokens, ct_answers, gt_answers)
            result_all_ct.append(result_ct)
            result_all_gt.append(result_gt)

        result_avg_ct = np.sum(result_all_ct, axis = 0)/20
        result_avg_gt = np.sum(result_all_gt, axis = 0)/20
        all_tuned_curve[r] = {'scale': scale, 'result_ct': result_avg_ct.tolist(), 'result_gt': result_avg_gt.tolist()}

        print(all_tuned_curve, flush = True)
   
    with open(f'{model_name}_{head}_tuned_curve_gt_m_camera_ready.json', 'w') as f:
        json.dump(all_tuned_curve, f)

run_all(5)