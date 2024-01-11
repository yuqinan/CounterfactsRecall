import argparse
import json
# Create an ArgumentParser object
parser = argparse.ArgumentParser()
#
## Define command-line arguments
parser.add_argument('--model', help='word to search')

parser.add_argument('--sample_file', default = 'pythia-2.8b_decoded.json', help='word to search')
parser.add_argument('--dataset', default = 'capital', help='word to search')
from pprint import pprint
#parser.add_argument('word', help='word to search')
#
args = parser.parse_args()

model_name = args.model
dataset = args.dataset

def one_scale():
    with open(f'pythia-1.4b_counterfacts_decoded.json', 'r') as f:
        ori = json.load(f)
        new = []
        for i in ori:
            new.append(i)
    if model_name == 'gpt2-xl':
        scale = ['-6.8_counterfacts_memory', '-10.0_counterfacts_icl', '10.0_counterfacts_memory','10.0_counterfacts_icl']
    if model_name == 'pythia-2.8b':
        scale = [10.0]
    if model_name == 'pythia-1.4b':
        #scale = ['3.6_capital_memory_camera_ready', '-1.5_capital_memory_camera_ready', '10.0', '-6.0']
        scale = [-1.5, 10.0, 1.5, -6.0]

    scale_all = {}

    for s in scale:  
        with open(f'{model_name}_intervened_{s}_counterfacts.json', 'r') as f:
            cur = json.load(f)

        al = []
        for i in cur:
            al.extend(i)

        scale_all[f'{s}'] = al
        print(len(al))
    for idx, i in enumerate(al):
        if idx<62992:
            for s in scale:
                i[f'intervene_decoded_{s}'] = scale_all[f'{s}'][idx]['intervened_decoded']
                new[idx] = i
    print(len(new))
    with open(f"{model_name}_counterfacts_decoded_all_scale_camera_ready.json", 'w') as f:
        json.dump(new, f)


def two_scale():
    with open(f'filter_counterfacts/{model_name}_clean_counterfacts.json', 'r') as f:
        ori = json.load(f)
    if model_name == 'pythia-1.4b':
        scale = [[-0.7, 10.0], [3.5, -6.0]]
    if model_name == 'pythia-2.8b':
        scale = [[-2.8, 10.0], [2.6, -1.5]]

    scale_all = {}
    for s in scale:  

        with open(f'/gpfs/data/epavlick/overwrite/{model_name}_intervened_{s[0]}_{s[1]}_counterfacts_both_heads.json', 'r') as f:
            cur = json.load(f)

        al = []
        for i in cur:
            al.extend(i)
        scale_all[f'{s[0]}_{s[1]}'] = al



    for idx, i in enumerate(al):
        for s in scale:
            i[f'intervene_decoded_{s[0]}_{s[1]}'] = scale_all[f'{s[0]}_{s[1]}'][idx]['intervened_decoded']
        ori[idx] = i
    print(len(ori))
    print(ori[-1])
    with open(f"{model_name}_decoded_both_heads_counterfacts.json", 'w') as f:
        json.dump(ori, f)

def two_scale():
    with open(f'decoded/{model_name}_decoded.json', 'r') as f:
        ori = json.load(f)
 
    if model_name == 'pythia-1.4b':
        scale = [[-0.7, 10.0], [3.5, -6.0]]
    if model_name == 'pythia-2.8b':
        scale = [[-2.8, 10.0], [2.6, -1.5]]

    scale_all = {}
    for s in scale:  
        with open(f'/gpfs/data/epavlick/overwrite/{model_name}_intervened_{s[0]}_{s[1]}_capital_both_heads.json', 'r') as f:
            cur = json.load(f)

        al = []
        for i in cur:
            al.extend(i)
        scale_all[f'{s[0]}_{s[1]}'] = al



    for idx, i in enumerate(al):
        if idx<62992:
            for s in scale:
                i[f'intervene_decoded_{s[0]}_{s[1]}'] = scale_all[f'{s[0]}_{s[1]}'][idx]['intervened_decoded']
            ori[idx] = i
    print(len(ori))
    print(ori[-1])
    with open(f"{model_name}_decoded_both_heads_capital.json", 'w') as f:
        json.dump(ori, f)

one_scale()