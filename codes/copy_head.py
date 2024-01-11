import json
with open('pythia-1.4b_decoded_all_scale.json', 'r') as f:
    data = json.load(f)

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

for idx, r in enumerate(data):
        if idx < 62992:
            ct_i, gt_i = check_answer(r['targets'][1], r['targets'][0], r['inputs'], r['intervene_decoded_10.0'])
            if r['gt_correct'] == 0 and r['correct'] ==0:
                 print(r['decoded'])
            #if r['gt_correct'] == 1 and ct_i == 1:
            #     print(r['decoded'])
            #     print(r[f'intervene_decoded_10.0'])
            #     print()