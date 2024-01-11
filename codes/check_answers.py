import json
import argparse
# Create an ArgumentParser object
parser = argparse.ArgumentParser()
#
## Define command-line arguments

parser.add_argument('--model', help='which set of models')
#
args = parser.parse_args()

model = args.model

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

with open(f'decoded/{model}_decoded.json', 'r') as f:
    decoded = json.load(f)
    
    for idx, i in enumerate(decoded):
        ct, gt = check_answer(i['targets'][1], i['targets'][0], i['inputs'], i['decoded'])
        i['correct'] = ct
        i['gt_correct'] = gt
        decoded[idx] = i

with open(f'decoded_with_answers/{model}_decoded_with_answers.json', 'w') as f:
    json.dump(decoded, f)
