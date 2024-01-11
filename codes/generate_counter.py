import json
import sys
sys.path.append('..')
import json
import torch
import argparse
from modeling import load_gpt2xl, load_pythia, GPT2Wrapper, PythiaWrapper
from transformer_lens import HookedTransformer, HookedTransformerConfig, FactoredMatrix, ActivationCache
device = "cuda" if torch.cuda.is_available() else "cpu"
parser = argparse.ArgumentParser()
parser.add_argument("model_name", help="name of model to be used")
args = parser.parse_args()
model_name = args.model_name

def generate_data():
    with open('../icl/few_shots.json', 'r') as f:
        original = json.load(f)
        clean = []
        for i in original:
            period = i['para_counter']['zero_shot'].index(".")
            clean.append({'ct':i['t_new'], 'gt': i['t_true'], "input_copy": i['copy_counter']['zero_shot'], 'input_para': i['para_counter']['zero_shot'][period+2: ]})
        print(len(clean))
        with open('counterfacts.json', 'w') as f:
            json.dump(clean, f)

generate_data()

def run_model(batch_size):
    model, tokenizer = load_gpt2xl()
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'left'
    with open(f'filter_counterfacts/{model_name}_clean_counterfacts.json', 'r') as f:
        all_facts = json.load(f)
    
    batch_prompts = []
    for i in range(0, len(all_facts), batch_size):
        batch = all_facts[i:i+batch_size]
        batch_prompts.append(batch)
    if i < len(all_facts) - batch_size:
        batch_prompts.append(all_facts[i:len(all_facts)])

    for idx, b in enumerate(batch_prompts):
        
        b_input = [i['input_para'] for i in b]

        prompt_ids = tokenizer(b_input, padding = True, return_tensors="pt")
        print(prompt_ids)
        prompt_ids['input_ids'] = prompt_ids['input_ids'].cuda()
        prompt_ids['attention_mask'] = prompt_ids['attention_mask'].cuda()
        g = model.generate(**prompt_ids, max_new_tokens = len(prompt_ids)+15, pad_token_id=tokenizer.eos_token_id, temperature = 0)
        decoded = [tokenizer.decode(i) for i in g]
        print(decoded)
        for i, d in enumerate(b):
            d['decoded'] = decoded[i]
            b[i] = d
        batch_prompts[idx] = b
    
    with open(f'{model_name}_decoded_counterfacts.json', 'w') as f:
        json.dump(batch_prompts, f)

def run_model_all():

    torch.set_grad_enabled(False)

    model = HookedTransformer.from_pretrained(model_name, device = device)
    with open(f'counterfacts.json', 'r') as f:
        all_facts = json.load(f)

    for idx, b in enumerate(all_facts):

        ct_tokens = model.to_tokens(b['input_para'], prepend_bos = False)
        print(b['input_para'])      

        decoded = model.tokenizer.decode(model.generate(ct_tokens, max_new_tokens = len(ct_tokens) + 15, temperature =0)[0])
        print(decoded)
        b['decoded'] = decoded
        all_facts[idx] = b

    with open(f'{model_name}_transformer_lense_decoded_counters.json', 'w') as f:
        json.dump(all_facts, f)
 
run_model(5)
#print("=============================================")
#generate_data()
#run_model_all()

#import re
#
#def edit_answer(path = f'{model_name}_counterfacts_decoded.json'):
#    with open(path, 'r') as f:
#        decoded  = json.load(f)
#        for idx, i in enumerate(decoded):
#            ct = 0
#            gt = 0
#            answer = i['decoded'][len(i['input_para']):]
#            if len(re.findall(i['ct'], answer)) > 0:
#                ct = 1
#            if len(re.findall(i['gt'], answer)) > 0:
#                gt = 1
#            i['ct_correct'] = ct
#            i['gt_correct'] = gt
#            decoded[idx] = i
#    
#    with open(f'edited_{path}', 'w') as f:
#        json.dump(decoded, f) 
#
#edit_answer()
#with open(f'edited_{model_name}_counterfacts_decoded.json', 'r') as f:
#    a = json.load(f)
#    print(len([i for i in a if i['ct_correct'] == 1]))
#    print(len([i for i in a if i['gt_correct'] == 1]))

#from pprint import pprint
#with open(f'pythia-1.4b_decoded_new.json', 'r') as f:
#    new_decoded = json.load(f)
#
#with open(f'pythia-1.4b_decoded.json', 'r') as f:
#    old_decoded = json.load(f)
#
#new_decoded = [i['decoded'] for i in new_decoded]
#old_decoded = [i['decoded'] for i in old_decoded]
#
#for i, d in enumerate(new_decoded):
#    if "The capital of South Korea is Minsk." in d:
#        print("============================")
#        print(d)
#        print(old_decoded[i])