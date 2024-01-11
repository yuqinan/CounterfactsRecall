import json
import argparse
# Create an ArgumentParser object
parser = argparse.ArgumentParser()
#
## Define command-line arguments

parser.add_argument('--model', help='which set of models')
parser.add_argument('--file', help='word to search')
#
args = parser.parse_args()

model = args.model
file = args.file

def check_answer(gt, ct, input, decoded_input):
    generated_answer_all = decoded_input[len(input):]
    if "Q:" not in generated_answer_all:
        generated_answer = generated_answer_all
    else:
         generated_answer = generated_answer_all[:generated_answer_all.index("Q:")]
    if gt in generated_answer:
        return 0, 1
    if ct in generated_answer:
        return 1, 0
    else:
        return 0, 0

import re   
def check_answer_counterfacts(i, intervene):
    ct = 0
    gt = 0
    try:
        a = i[intervene].index(i['input_para'])
    
        answer = i[intervene][a+len(i['input_para']):]

        if len(re.findall(i['ct'], answer)) > 0:
            ct = 1
        if len(re.findall(i['gt'], answer)) > 0:
            gt = 1
    except ValueError:
        print()
        return 3, 3

    return ct, gt

def generate_answers():
    with open(f"{model}_decoded_all_scale_camera_ready.json" ,'r') as f:
        decoded = json.load(f)
        if model == 'pythia-1.4b':
            #scale = ['3.6_capital_memory_camera_ready', '-1.5_capital_memory_camera_ready', '10.0', '-6.0']
            scale =  [-1.5, 10.0, 1.5, -6.0]

        if model == 'pythia-2.8b':
            scale = [-1.5, -2.8, 2.6, 10.0]
        if model == 'gpt2-xl':
            scale = ['-6.8_capital', '-10.0_capital', '10.0_capital_memory','10.0_capital_icl']
        for idx, r in enumerate(decoded):
            print(r)
            if idx < 62992:
                for s in scale:
                    ct, gt = check_answer(r['targets'][1], r['targets'][0], r['inputs'], decoded[idx]['decoded'])
                    ct_i, gt_i = check_answer(r['targets'][1], r['targets'][0], r['inputs'], r[f'intervene_decoded_{s}'])
                    r['correct'] = ct
                    r['gt_correct'] = gt
                    r[f'intervened_correct_{s}'] = ct_i
                    r[f'intervened_gt_correct_{s}'] = gt_i
                
                decoded[idx] = r

        with open(f'final_{model}_camera_ready.json', 'w') as f:
            json.dump(decoded[:62992], f)
    
def generate_answers_counterfacts():
    with open(f"{model}_counterfacts_decoded_all_scale_camera_ready.json" ,'r') as f:
        decoded = json.load(f)
        print(len(decoded))
        if model == 'pythia-1.4b':
            scale = [1.5, -1.5, 10.0, -6.0]
        if model == 'pythia-2.8b':
            scale = [10.0]#[-1.5, -2.8, 2.6]
        if model == 'gpt2-xl':
            scale = ['-6.8_counterfacts_memory', '-10.0_counterfacts_icl', '10.0_counterfacts_memory','10.0_counterfacts_icl']
        for idx, r in enumerate(decoded):
                for s in scale:
                    ct, gt = check_answer_counterfacts(r, 'decoded')
                    ct_i, gt_i = check_answer_counterfacts(r, f'intervene_decoded_{s}')
                    r['correct'] = ct
                    r['gt_correct'] = gt
                    r[f'intervened_correct_{s}'] = ct_i
                    r[f'intervened_gt_correct_{s}'] = gt_i
                
                decoded[idx] = r

        with open(f'final_{model}_counterfacts_camera_ready.json', 'w') as f:
            json.dump(decoded, f)

def generate_answers_counterfacts_both():
    with open(f"{model}_decoded_both_heads_capital.json" ,'r') as f:
        decoded = json.load(f)
        print(len(decoded))
        if model == 'pythia-1.4b':
            scale = [[-0.7, 10.0], [3.5, -6.0]]
        if model == 'pythia-2.8b':
            scale = [[-2.8, 10.0], [2.6, -1.5]]
        for idx, r in enumerate(decoded):
                for s in scale:
                    ct, gt = check_answer(r['targets'][1], r['targets'][0], r['inputs'], decoded[idx]['decoded'])
                    ct_i, gt_i = check_answer(r['targets'][1], r['targets'][0], r['inputs'], r[f'intervene_decoded_{s[0]}_{s[1]}'])
                    r['correct'] = ct
                    r['gt_correct'] = gt
                    r[f'intervened_correct_{s[0]}_{s[1]}'] = ct_i
                    r[f'intervened_gt_correct_{s[0]}_{s[1]}'] = gt_i
                
                decoded[idx] = r

        with open(f'final_{model}_capitals_both.json', 'w') as f:
            json.dump(decoded, f)

       
from pprint import pprint
generate_answers_counterfacts()

with open(f'final_{model}_counterfacts_camera_ready.json', 'r') as f:
    a = json.load(f)
    pprint(a[-1])