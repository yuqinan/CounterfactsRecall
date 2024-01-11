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
    with open('counterfacts.json', 'r') as f:
        original = json.load(f)
        clean = []
        for idx, i in enumerate(original):
            try:
                period = i['input_para'].index(".")
            except Exception:
                print(i['input_para'])
            i['input_para']= i['input_para'][period+2: ]
            original[idx] = i
        print(len(clean))
        with open('counterfacts_no_counter.json', 'w') as f:
            json.dump(original, f)

#generate_data()

def run_model(batch_size):
    model, tokenizer = load_gpt2xl()
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'left'
    with open('counterfacts_no_counter.json', 'r') as f:
        all_facts = json.load(f)
       
    batch_prompts = []
    for i in range(0, len(all_facts), batch_size):
        batch = all_facts[i:i+batch_size]
        batch_prompts.append(batch)
    if i < len(all_facts) - batch_size:
        batch_prompts.append(all_facts[i:len(all_facts)])

    print(batch_prompts)
    for idx, b in enumerate(batch_prompts):
        
        b_input = [i['input_para'] for i in b]

        prompt_ids = tokenizer(b_input, padding=True, return_tensors="pt")
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

    with open(f'/gpfs/data/epavlick/overwrite/{model_name}_decoded_without_counters.json', 'w') as f:
        json.dump(all_facts, f)

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

    with open(f'{model_name}_transformer_lense_decoded_without_counters.json', 'w') as f:
        json.dump(all_facts, f)
 

with open(f'/gpfs/data/epavlick/overwrite/{model_name}_decoded_without_counters.json', 'r') as f:
    data = json.load(f)
    print(len(data))
    clean = []

with open(f'gpt2-xl_decoded_counterfacts.json', 'r') as f:
    data = json.load(f)
    ori = []
    for d in data:
        ori.append(d)
    print(len(ori))

for idx, i in enumerate(data):
    i['check_decode'] = i['decode']
    i['decode'] = ori[idx]['decode']
    data[idx] = i


with open(f'/gpfs/data/epavlick/overwrite/{model_name}_decoded_without_counters.json', 'r') as f:
    data = json.load(f)
    print(len(data))
    clean = []

with open('counterfacts.json', 'r') as f:
    original = json.load(f)

    for idx, i in enumerate(data):
        try:
            c = i['decoded'].index(i['input_para'])+len(i['input_para'])
        except ValueError:
            continue
        answer = i['decoded'][c-1:]
        if i['gt'] in answer:
            i['input_para'] = original[idx]['input_para']
            clean.append(data[idx])
    print(len(clean))
with open(f'filter_counterfacts/{model_name}_clean_counterfacts.json', 'w') as f:
    json.dump(clean, f)

#with open(f'/gpfs/data/epavlick/overwrite/{model_name}_decoded_without_counters.json', 'r') as f:
#    data = json.load(f)
#    clean = []
#    for idx, i in enumerate(data):
#        if i['gt'] in i['decoded']:
#            clean.append(d[idx])
#with open(f'filter_counterfacts/{model_name}_clean_counterfacts.json', 'w') as f:
#    json.dump(clean, f)

