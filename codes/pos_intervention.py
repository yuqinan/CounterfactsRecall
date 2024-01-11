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

parser.add_argument('--sample_file', default = 'pythia-1.4b_decoded.json', help='word to search')
from pprint import pprint
#parser.add_argument('word', help='word to search')
#
args = parser.parse_args()



import random

def sample(input, pick = 100):
    return [input[random.choice(range(len(input)))] for _ in range(pick)]

all_rank = {}
for i in range(10):
    all_rank[i] = []

def form_subset_gt(model_name = args.model):
    with open(f'{model_name}_abs_icl_country_all_decoded.json', 'r') as f:
        data = json.load(f)  
        for k, v in data.items():
            v = [i for i in v if i['correct']==1]
            data[k] = v     
    with open(f'{model_name}_ct_correct.json', 'w') as f:
        json.dump(data, f)


form_subset_gt()

with open(f'{args.model}_ct_correct.json', 'r') as f:
    all_rank = json.load(f)

for k, v in all_rank.items():
    if len(v)< 10:
        end = len(v)
    else:
        end = 10
    sampled= []
    while len(sampled) < end:
        sampled.append(v[random.choice(range(len(v)))])
    all_rank[k] = sampled
    


torch.set_grad_enabled(False)
device = "cuda" if torch.cuda.is_available() else "cpu"

model_name = args.model

model = HookedTransformer.from_pretrained(model_name, device = device)

#print("----------------------------------------------------------")
#ct_text = 'The capital of Cambodia is Pago Pago.\nQ: What is the capital of Cambodia?\nA: It is'
#ct_tokens = model.to_tokens(ct_text)
#ct_str_tokens = model.to_str_tokens(ct_text)
#ct_logits, ct_cache = model.run_with_cache(ct_tokens)
#print(model.tokenizer.decode(model.generate(ct_tokens, temperature =0)[0]))
# answers = [[' '+model.tokenizer.tokenize(sample['targets'][0])[0][1:],
#             ' '+model.tokenizer.tokenize(sample['targets'][1])[0][1:]] for sample in sample_gt]
# print(answers_tokens)
# # List of the token (ie an integer) corresponding to each answer, in the format (correct_token, incorrect_token)
# answer_tokens = []
# for i in range(len(prompts)):
#     answer_tokens.append(
#         (
#             model.to_single_token(answers[i][0]),
#             model.to_single_token(answers[i][1])
#         ))
        

# answer_tokens = torch.tensor(answer_tokens).cuda()
#ct_tokens = model.to_tokens(ct_text)
#ct_str_tokens = model.to_str_tokens(ct_text)
#ct_logits, ct_cache = model.run_with_cache(ct_tokens)

#answer_residual_directions = model.tokens_to_residual_directions(answer_tokens)
#print("Answer residual directions shape:", answer_residual_directions.shape)
#logit_diff_directions = answer_residual_directions[:, 0] - answer_residual_directions[:, 1]
#print("Logit difference directions shape:", logit_diff_directions.shape)

# Move the tokens to the GPU


#answer_residual_directions = model.tokens_to_residual_directions(torch.tensor([(model.to_single_token(" Vienna"), model.to_single_token(" Tir"))]))
#logit_diff_directions = answer_residual_directions[:, 0] - answer_residual_directions[:, 1]
#print("Logit difference directions shape:", logit_diff_directions.shape)

# cache syntax - resid_post is the residual stream at the end of the layer, -1 gets the final layer. The general syntax is [activation_name, layer_index, sub_layer_type]. 
#final_residual_stream = ct_cache["resid_post", -1]
#print("Final residual stream shape:", final_residual_stream.shape)
#final_token_residual_stream = final_residual_stream[:, -1, :]
## Apply LayerNorm scaling
## pos_slice is the subset of the positions we take - here the final token of each prompt
#scaled_final_token_residual_stream = ct_cache.apply_ln_to_stack(final_token_residual_stream, layer = -1, pos_slice=-1)
#
#average_logit_diff = einsum("batch d_model, batch d_model -> ", scaled_final_token_residual_stream, logit_diff_directions)
#print("Calculated average logit diff:", average_logit_diff.item())

# cache syntax - resid_post is the residual stream at the end of the layer, -1 gets the final layer. The general syntax is [activation_name, layer_index, sub_layer_type]. 
#final_residual_stream = cache["resid_post", -1]
#
#print("Final residual stream shape:", final_residual_stream.shape)
#
### 
#final_token_residual_stream = final_residual_stream[:, -1, :]
# Apply LayerNorm scaling
# pos_slice is the subset of the positions we take - here the final token of each prompt


# def visualize_attention_patterns(
#     heads, 
#     local_cache = None, 
#     local_tokens =None, 
#     title = ""):
#     # Heads are given as a list of integers or a single integer in [0, n_layers * n_heads)
#     if isinstance(heads, int):
#         heads = [heads]
#     elif isinstance(heads, list) or isinstance(heads, torch.Tensor):
#         heads = utils.to_numpy(heads)
    
#     labels = []
#     patterns = []
#     batch_index = 0
#     for head in heads:
#         layer = head // model.cfg.n_heads
#         head_index = head % model.cfg.n_heads
#         # Get the attention patterns for the head
#         # Attention patterns have shape [batch, head_index, query_pos, key_pos]
#         patterns.append(local_cache["attn", layer][batch_index, head_index])
#         labels.append(f"L{layer}H{head_index}")
#     str_tokens = model.to_str_tokens(local_tokens)
#     patterns = torch.stack(patterns, dim=-1)


def get_data(sample_gt, rank, batch_size):
    
    data = sample_gt[rank]
    batch_data = []
    for i in range(0, len(data), batch_size):
        batch = data[i:i+batch_size]
        batch_data.append(batch)
    batch_data.append(data[i:len(data)])

    prompts= [sample['inputs'] for sample in sample_gt[rank]]
    batch_prompts = []
    for i in range(0, len(prompts), batch_size):
        batch = prompts[i:i+batch_size]
        batch_prompts.append(batch)
    batch_prompts.append(prompts[i:len(prompts)])

    
    icl_answers = [sample['targets'][1] for sample in sample_gt[rank]]
    batch_icl_answers = []
    for i in range(0, len(icl_answers), batch_size):
        batch = icl_answers[i:i+batch_size]
        batch_icl_answers.append(batch)
    batch_icl_answers.append(icl_answers[i:len(icl_answers)])

    
    answer_tokens = torch.stack([torch.stack([model.to_tokens(sample['targets'][1])[0][1],
                model.to_tokens(sample['targets'][1])[0][1]]) for sample in sample_gt[rank]])
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

def residual_stack_to_logit_diff(logit_diff_directions, residual_stack, cache):
    scaled_residual_stack = cache.apply_ln_to_stack(residual_stack, layer = -1, pos_slice=-1)
    return einsum("... batch d_model, batch d_model -> ...", scaled_residual_stack, logit_diff_directions)


from pprint import pprint
def get_heads(logits_diff):
    # Convert the 1D index to 2D indices
    l = logits_diff.shape[1]
    diff = logits_diff.flatten()
    max_index = torch.topk(diff, k = 3)
    min_index = torch.topk(diff, k = 3, largest = False)
    # Convert the 1D index to 2D indices
    return {'pos': [[i//l, i% l] for i in max_index[1]],
            'neg': [[i//l, i% l] for i in min_index[1]]}


def head_upweight_hook(
    value,#: Float[torch.Tensor, "batch pos head_index d_head"],
    hook: HookPoint,
    head_idx,
    scale
):# -> Float[torch.Tensor, "batch pos head_index d_head"]:
    value[:, :, head_idx, :] *=scale
    return value

def change_weights(input,token,  pos_scale, neg_scale):
    model.reset_hooks()
    fwd_hooks = []
    for idx, i in enumerate(input['pos']):
        upweight_hook = partial(head_upweight_hook, head_idx=i[1], scale= pos_scale[idx]) #specfic memory
        fwd_hooks.append((utils.get_act_name('v', i[0]), upweight_hook))
    for idx, i in enumerate(input['neg']):
        downweight_hook = partial(head_upweight_hook, head_idx=11, scale= neg_scale[idx]) #Albania
        fwd_hooks.append((utils.get_act_name('v', 11), downweight_hook))#upweight, attends to counterfactual
   
    weighted_logits = model.run_with_hooks(token,fwd_hooks=fwd_hooks)  

    return weighted_logits

def generate_weighted_output(input, token,  pos_scale, neg_scale):
    model.reset_hooks()
    for idx, i in enumerate(input['pos']):
        upweight_hook = partial(head_upweight_hook, head_idx=i[1], scale= pos_scale[idx])
        model.blocks[i[0]].attn.hook_v.add_hook(upweight_hook)
    for idx, i in enumerate(input['neg']):
        downweight_hook = partial(head_upweight_hook, head_idx=11, scale= neg_scale[idx]) #Albania
        model.blocks[11].attn.hook_v.add_hook(downweight_hook)

    return model.tokenizer.decode(model.generate(token,  max_new_tokens = len(token)+15, temperature =0)[0])
    

def logits_to_logit_diff(logits, correct_answer=" Bel", incorrect_answer=" Vienna"):
    # model.to_single_token maps a string value of a single token to the token index for that token
    # If the string is not a single token, it raises an error.
    correct_index = model.to_single_token(correct_answer)
    incorrect_index = model.to_single_token(incorrect_answer)
    return logits[0, -1, correct_index] - logits[0, -1, incorrect_index]

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
        all_difference.append(per_head_logit_diffs)
        all_logit.append(original_logits)
        all_cache.append(cache)
    all_difference = torch.stack(all_difference)
    all_difference = torch.mean(all_difference, dim = 0)
    input = get_heads(all_difference)  

    return input

def intervention(tokens, icl_answers, input, pos_scale, neg_scale):
    #import random
    #l = range(24)
    #h = range(16)
    #
    #layers = random.sample(l, 6)
    #heads = random.sample(h,6)
    #
    #input = {'pos': [[layers[0], heads[0]], [layers[1], heads[1]], [layers[2], heads[2]]],
    #         'neg': [[layers[3], heads[3]], [layers[4], heads[4]], [layers[5], heads[5]]]}
    evaluate = []
    for idx, d in enumerate(tokens):

        #weighted_logits = change_weights(input, tokens[idx], pos_scale, neg_scale)

        #print(logits_to_logit_diff(all_logit[idx], answers[idx][0], answers[idx][1]))
        #print(logits_to_logit_diff(weighted_logits, answers[idx][0], answers[idx][1]))
        #a = weighted_logits[0,-1].argsort(descending=True)[:20]
        #b = original_logits[0,-1].argsort(descending=True)[:20]


        #print(model.tokenizer.decode(b))
        #print()
        #print(model.tokenizer.decode(a))

        a = generate_weighted_output(input, d,  pos_scale, neg_scale)

        pprint(a)
        model.reset_hooks()
        import re
        if len(re.findall(icl_answers[idx],a))>0:         
            evaluate.append(1)
        else:
            #print(model.tokenizer.decode(model.generate(tokens[idx],  max_new_tokens = 30, temperature =0)[0]))
            evaluate.append(0)
            #print(a)     

    return np.sum(evaluate)

    

def tune_curve_freq(tokens, icl_answers, input):
    result = []
    scale = []
    for i in np.arange(1, 10, 0.5):
        print(i)
        scale.append(i)
        acc = intervention(tokens, icl_answers, input, [1, 1, 1], [i, 1, 1])
        result.append(acc)

        print(scale, result, flush = True)
    
    return scale, result

def run_all(batch_size):
    all_tuned_curve = {}
    heads = []
    for r in range(10):
        result_all = []
        batch_data, batch_prompts, batch_tokens, batch_icl_answers, batch_logit_diff_directions = get_data(all_rank, str(r), batch_size)
        for idx, i in enumerate(batch_prompts):
            data, prompts, tokens, icl_answers, logit_diff_directions = batch_data[idx], batch_prompts[idx], batch_tokens[idx], batch_icl_answers[idx], batch_logit_diff_directions[idx]
            input = get_heads_id(tokens, logit_diff_directions)
            print(data,flush = True)
            print(input, flush = True)
            heads.append(input)
            scale, result =  tune_curve_freq(tokens, icl_answers, input)
            result_all.append(result)

        result_avg = np.sum(result_all, axis = 0)/10
        all_tuned_curve[r] = {'scale': scale, 'result': result_avg.tolist()}
        print(all_tuned_curve, flush = True)
    
    with open(f'{model_name}_pos_tuned_curve.json', 'w') as f:
        json.dump(all_tuned_curve, f)
    #with open(f'{model_name}_pos_heads.json', 'w') as f:
    #    json.dump(heads, f)

def residual_stack_to_logit_diff(logit_diff_directions, residual_stack, cache):
    scaled_residual_stack = cache.apply_ln_to_stack(residual_stack, layer = -1, pos_slice=-1)
    return einsum("... batch d_model, batch d_model -> ...", scaled_residual_stack, logit_diff_directions)


############################ Here for decode ###############################
def intervention_decode(tokens, input):
    decode = []
    for idx, d in enumerate(tokens):
        a = generate_weighted_output(input, d,  [1, 1, 1], [-2.8, 1, 1])
        pprint(a)
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


def run_all_onefile(batch_size, fp):
    #with open(f'{model_name}_decoded.json', 'r') as f:
    #    data = json.load(f)
    #    data = [i for i in data if i['gt_correct'] == 1]
#
    #with open('temp.json', 'w') as f:
    #    json.dump(data, f)

    all_intervened_data = []
    heads = []
    batch_data, batch_prompts, batch_tokens, batch_icl_answers, batch_logit_diff_directions = get_data_one_file(batch_size, fp)
    for idx, i in enumerate(batch_prompts):
        data, prompts, tokens, icl_answers, logit_diff_directions = batch_data[idx], batch_prompts[idx], batch_tokens[idx], batch_icl_answers[idx], batch_logit_diff_directions[idx]
        input = get_heads_id(tokens, logit_diff_directions)
        print()
        input['pos'] = [[i[0].cpu().item(), i[1].cpu().item()] for i in input['pos']]
        input['neg'] = [[i[0].cpu().item(), i[1].cpu().item()] for i in input['neg']]
        print(input, flush = True)
        heads.append(input)
        decoded =  intervention_decode(tokens, input)
        print(decoded, flush = True)
        for id, d in enumerate(data):
            d['intervened_decoded'] = decoded[id]
            data[id] = d
        all_intervened_data.append(data)
    

    with open(f"{model_name}_intervened_data.json", 'w') as f:
        json.dump(all_intervened_data, f)


run_all(4)
