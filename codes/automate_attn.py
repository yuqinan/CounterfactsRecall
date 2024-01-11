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

parser.add_argument('--sample_file', default = 'pythia-2.8b_decoded.json', help='word to search')
parser.add_argument('--dataset', default = 'capital', help='word to search')
parser.add_argument('--head', default = 'memory', help='word to search')
parser.add_argument('--scale', default = '1', help='word to search')
from pprint import pprint
#parser.add_argument('word', help='word to search')
#
args = parser.parse_args()

import random

torch.set_grad_enabled(False)
device = "cuda" if torch.cuda.is_available() else "cpu"

model_name = args.model
head = args.head
scale = args.scale
scale = float(scale)
dataset = args.dataset

model = HookedTransformer.from_pretrained(model_name, device = device)

def head_upweight_hook(
    value,#: Float[torch.Tensor, "batch pos head_index d_head"],
    hook: HookPoint,
    head_idx,
    scale
):# -> Float[torch.Tensor, "batch pos head_index d_head"]:
    value[:, :, head_idx, :] *=scale
    return value


def generate_weighted_output(token, scale):
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
            downweight_hook = partial(head_upweight_hook, head_idx=20, scale= scale) ## icl head
            model.blocks[29].attn.hook_v.add_hook(downweight_hook) ## scale up (positive) -> icl ## scale down (negative) -> gt

    return model.tokenizer.decode(model.generate(token,  max_new_tokens = len(token)+15, temperature =0)[0])
    

############################ Here for decode ###############################
def intervention_decode(tokens, scale):
    decode = []
    for idx, d in enumerate(tokens):
        a = generate_weighted_output(d, scale)
        model.reset_hooks()
        decode.append(a)

    return decode


def get_data_one_file(batch_size, path = args.sample_file):

    with open(path, 'r') as f:
        data = json.load(f)

    batch_data = []
    for i in range(0, len(data), batch_size):
        batch = data[i:i+batch_size]
        batch_data.append(batch)
    batch_data.append(data[i:len(data)])

    prompts= [sample['inputs'] for sample in data]
    batch_prompts = []
    for i in range(0, len(prompts), batch_size):
        batch = prompts[i:i+batch_size]
        batch_prompts.append(batch)
    batch_prompts.append(prompts[i:len(prompts)])

    
    icl_answers = [sample['targets'][0] for sample in data]
    batch_icl_answers = []
    for i in range(0, len(icl_answers), batch_size):
        batch = icl_answers[i:i+batch_size]
        batch_icl_answers.append(batch)
    batch_icl_answers.append(icl_answers[i:len(icl_answers)])

    
    answer_tokens = torch.stack([torch.stack([model.to_tokens(sample['targets'][0])[0][1],
                model.to_tokens(sample['targets'][1])[0][1]]) for sample in data])
    batch_answer_tokens = []
    for i in range(0, len(answer_tokens), batch_size):
        batch = answer_tokens[i:i+batch_size]
        batch_answer_tokens.append(batch)
    batch_answer_tokens.append(answer_tokens[i: len(answer_tokens)])
    
    
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
    batch_tokens.append(tokens[i:len(tokens)])

    return batch_data, batch_prompts, batch_tokens, batch_icl_answers, batch_logit_diff_directions


def get_data_counterfacts(batch_size, path = args.sample_file):

    with open(path, 'r') as f:
        d= json.load(f)
    data = []
    for i in d:
        data.append(i)
    print(len(data))


    batch_data = []
    for i in range(0, len(data), batch_size):
        batch = data[i:i+batch_size]
        batch_data.append(batch)
    prompts= [sample['input_para'] for sample in data]
    batch_prompts = []
    for i in range(0, len(prompts), batch_size):
        batch = prompts[i:i+batch_size]
        batch_prompts.append(batch)

    
    icl_answers = [sample['ct'] for sample in data]
    batch_icl_answers = []
    for i in range(0, len(icl_answers), batch_size):
        batch = icl_answers[i:i+batch_size]
        batch_icl_answers.append(batch)

    
    answer_tokens = torch.stack([torch.stack([model.to_tokens(sample['ct'])[0][1],
                model.to_tokens(sample['ct'])[0][1]]) for sample in data])
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

    return batch_data, batch_prompts, batch_tokens, batch_icl_answers, batch_logit_diff_directions

from tqdm import tqdm
def run_all_onefile(batch_size, fp):
    all_intervened_data = []
    if dataset == 'capital':
        batch_data, batch_prompts, batch_tokens, batch_icl_answers, batch_logit_diff_directions = get_data_one_file(batch_size, fp)
    else:
        batch_data, batch_prompts, batch_tokens, batch_icl_answers, batch_logit_diff_directions = get_data_counterfacts(batch_size, fp)
    for idx, i in tqdm(enumerate(batch_prompts)):
        data, prompts, tokens, icl_answers, logit_diff_directions = batch_data[idx], batch_prompts[idx], batch_tokens[idx], batch_icl_answers[idx], batch_logit_diff_directions[idx]
        decoded =  intervention_decode(tokens, scale)
        for id, d in enumerate(data):
            d['intervened_decoded'] = decoded[id]
            pprint(decoded[id])
            pprint(d['decoded'])
            data[id] = d
        all_intervened_data.append(data)
    
    

    with open(f"{model_name}_intervened_{scale}_{dataset}_{head}_camera_ready.json", 'w') as f:
        json.dump(all_intervened_data, f)

if dataset == 'capital':
    run_all_onefile(4, f'decoded/{model_name}_decoded.json')
else:
    run_all_onefile(4, f'pythia-1.4b_counterfacts_decoded.json')
