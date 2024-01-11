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
args = parser.parse_args()

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
            if pick['gt_correct'] == 1:
                sampled.append(pick)
        all_rank[k] = sampled
    with open(f'{model_name}_tuning_set_gt_head.json', 'w') as f:
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
            if pick['correct'] == 1:
                sampled.append(pick)
        all_rank[k] = sampled
    with open(f'{model_name}_tuning_set_ct_head.json', 'w') as f:
        json.dump(all_rank, f)
        
get_tuning_data_gt()
get_tuning_data_ct()

def get_data(sample_gt, rank, batch_size):
    
    #data = sample_gt[rank]
    data = sample_gt
    batch_data = []
    for i in range(0, len(data), batch_size):
        batch = data[i:i+batch_size]
        batch_data.append(batch)

    prompts= [sample['inputs'] for sample in sample_gt]
    batch_prompts = []
    for i in range(0, len(prompts), batch_size):
        batch = prompts[i:i+batch_size]
        batch_prompts.append(batch)

    
    ct_answers = [sample['targets'][0] for sample in sample_gt]
    batch_ct_answers = []
    for i in range(0, len(ct_answers), batch_size):
        batch = ct_answers[i:i+batch_size]
        batch_ct_answers.append(batch)


    gt_answers = [sample['targets'][1] for sample in sample_gt]
    batch_gt_answers = []
    for i in range(0, len(gt_answers), batch_size):
        batch = gt_answers[i:i+batch_size]
        batch_gt_answers.append(batch)
    

    answer_tokens = torch.stack([torch.stack([model.to_tokens(sample['targets'][0])[0][1],
                model.to_tokens(sample['targets'][1])[0][1]]) for sample in sample_gt])
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
    
    return batch_data, batch_prompts, batch_tokens, batch_ct_answers, batch_logit_diff_directions

def get_heads(logits_diff):
    # Convert the 1D index to 2D indices
    l = logits_diff.shape[1]
    diff = logits_diff.flatten()
    max_index = torch.topk(diff, k = 3)
    min_index = torch.topk(diff, k = 3, largest = False)
    # Convert the 1D index to 2D indices
    return {'pos': [[i//l, i% l] for i in max_index[1]],
            'neg': [[i//l, i% l] for i in min_index[1]]}


def residual_stack_to_logit_diff(logit_diff_directions, residual_stack, cache):
    scaled_residual_stack = cache.apply_ln_to_stack(residual_stack, layer = -1, pos_slice=-1)
    return einsum("... batch d_model, batch d_model -> ...", scaled_residual_stack, logit_diff_directions)


def get_heads_id(tokens, logit_diff_directions):
    all_difference = []
    all_logit = []
    all_cache = []
    for idx, t in enumerate(tokens):
        original_logits, cache = model.run_with_cache(t)
        final_token_residual_stream = cache["resid_post", -1][:, -1, :]
        scaled_final_token_residual_stream = cache.apply_ln_to_stack(final_token_residual_stream, layer = -1, pos_slice=-1)
        #average_logit_diff = einsum("batch d_model, batch d_model -> ", scaled_final_token_residual_stream, logit_diff_directions)
        per_head_residual, labels = cache.stack_head_results(layer=-1, pos_slice=-1, return_labels=True)
        per_head_logit_diffs = residual_stack_to_logit_diff(logit_diff_directions, per_head_residual, cache)
        per_head_logit_diffs = einops.rearrange(per_head_logit_diffs, "(layer head_index) -> layer head_index", layer=model.cfg.n_layers, head_index=model.cfg.n_heads)
        all_difference.append(per_head_logit_diffs.tolist())
        all_logit.append(original_logits)
        all_cache.append(cache)
    with open('headmap.json', 'w') as f:
        json.dump(all_difference, f)
    #all_difference = torch.stack(all_difference)
    all_difference = torch.mean(torch.tensor(all_difference), dim = 0)
    input = get_heads(all_difference)  

    return all_difference, input

def get_head(batch_size):
    with open(f'sample_list_more.json', 'r') as f:
        all_rank= json.load(f)
        
    head = {'gt':[], 'ct':[]}
    result_all = []
    batch_data, batch_prompts, batch_tokens, batch_icl_answers, batch_logit_diff_directions = get_data(all_rank, '0', batch_size)
    for idx, i in enumerate(batch_prompts):
        data, prompts, tokens, icl_answers, logit_diff_directions = batch_data[idx], batch_prompts[idx], batch_tokens[idx], batch_icl_answers[idx], batch_logit_diff_directions[idx]
        # input = get_heads_id(tokens, logit_diff_directions)
        all_difference, input = get_heads_id(tokens, logit_diff_directions)
        input['pos'] = [[i[0].item(), i[1].item()] for i in input['pos']]
        input['neg'] = [[i[0].item(), i[1].item()] for i in input['neg']]
        head['ct'].append(input)
        print(input, flush = True)
    print(all_difference)

    #with open(f'{model_name}_tuning_set_gt_head.json', 'r') as f:
    #    all_rank = json.load(f)
    #for r in range(10):
    #    result_all = []
    #    batch_data, batch_prompts, batch_tokens, batch_icl_answers, batch_logit_diff_directions = get_data(all_rank, str(r), batch_size)
    #    for idx, i in enumerate(batch_prompts):
    #        data, prompts, tokens, icl_answers, logit_diff_directions = batch_data[idx], batch_prompts[idx], batch_tokens[idx], batch_icl_answers[idx], batch_logit_diff_directions[idx]
    #        input = get_heads_id(tokens, logit_diff_directions)
    #        input['pos'] = [[i[0].item(), i[1].item()] for i in input['pos']]
    #        input['neg'] = [[i[0].item(), i[1].item()] for i in input['neg']]
    #        head['gt'].append(input)
    #        print(input, flush = True)
#
    #with open(f'{model_name}_head.json', 'w') as f:
    #    json.dump(head, f)
    
get_head(20)
