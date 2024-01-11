import csv
import argparse
import csv
import re
import pandas as pd
import json
import numpy as np
from tqdm import tqdm
from pprint import pprint

# Create an ArgumentParser object
parser = argparse.ArgumentParser()
#
## Define command-line arguments

parser.add_argument('--model', help='which set of models')
#parser.add_argument('word', help='word to search')
#
args = parser.parse_args()

model = args.model


def add_intervention_answer(path = 'pythia-1.4b_intervened_3.6_capital_memory_camera_ready.json'):
    with open(path, 'r') as f:
        intervention = json.load(f)
    add_answer = []
    all_flat = []
    for i in intervention:
        all_flat.extend(i)
    for i in all_flat:
        correct = 0
        gt_correct = 0
        targets = i['targets']
        decoded = i['intervened_decoded']
        if len(re.findall(targets[0],decoded))>1:
            correct = 1
        if targets[1] in decoded:
            if decoded.index(targets[1]) > decoded.index('A:'):
                gt_correct = 1
        i['intervened_correct'] = correct
        i['intervened_gt_correct'] = gt_correct
        add_answer.append(i)
    with open('pythia-1.4b_intervened_3.6_capital_memory_camer_ready_data_edited.json', 'w') as f2:
        json.dump(add_answer, f2)
    


def read_csv_as_dict(filename):
    data = []
    with open(filename, 'r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            data.append(dict(row))
    return data

def p(lsort, input, criteria):   
    import numpy as np

    # Calculate percentiles
    percentiles = np.percentile(lsort, range(0, 101, 10))

    # Break the list into sublists
    sublists = []
    for i in range(len(percentiles) - 1):
        if criteria == "abs_capital":
            sublist = [x['capital'] for x in input if percentiles[i] <= float(x[criteria]) <= percentiles[i + 1]]
        if criteria == "bi_gt_country":
            sublist = [x['capital'] for x in input if percentiles[i] <= float(x[criteria]) <= percentiles[i + 1]]
        
        if criteria == 'abs_icl_country':
            sublist = [x['country'] for x in input if percentiles[i] <= float(x[criteria]) <= percentiles[i + 1]]
        if criteria == 'bi_gt_capital':
            sublist = [x['country'] for x in input if percentiles[i] <= float(x[criteria]) <= percentiles[i + 1]]

        sublists.append(list(set(sublist)))

    for i, sublist in enumerate(sublists):
        print(f'---- sublist {i} -------')
        pprint(sublist)

    return sublists

def generate_occ_for_one_model(name, criteria):

    data = read_csv_as_dict(f"{name}_big_results_decoded.csv")
    occ = [float(d[criteria]) for d in data]

    if "pairs" not in criteria:
        return list(set(occ)), data   
     
    return occ, data


def generate_samples(criteria, path = f"{model}_memory_neg.json"):

    with open(path, 'r') as file:
        all_result = json.load(file)

    lsort, input = generate_occ_for_one_model(model, criteria)
    sublists = p(lsort, input, criteria)
    ranked_result = {}

    for i in range(10):
        ranked_result[i] = []


    for data in all_result:
        for rank, cp in enumerate(sublists):
                c = data['targets'][2][1:].lower()
                if "abs_capital" == criteria or "bi_gt_country" == criteria:
                    if data['targets'][0][1:].lower() in cp:
                        ranked_result[rank].append(data)
                if 'abs_icl_country' == criteria or 'bi_gt_capital' == criteria:
                    if c in cp:
                        ranked_result[rank].append(data)
    
    with open(f'{model}_{criteria}_all_decoded.json', 'w') as f:
        json.dump(ranked_result, f)




def generate_onc_cri(criteria):

    generate_samples(criteria)

    with open(f'{model}_{criteria}_all_decoded.json', 'r') as f:
        data = json.load(f)
    
    result = {'gt': [], 'ct': [], 'i_gt': [], 'i_ct': []}
    
    for k, v in data.items():
        result['gt'].append(len([i for i in v if i['gt_correct'] == 1]))
        result['ct'].append(len([i for i in v if i['correct'] == 1]))
        result['i_ct'].append(len([i for i in v if i['intervened_correct'] == 1]))
        result['i_gt'].append(len([i for i in v if i['intervened_gt_correct'] == 1]))
                
    return result

def generate_all_cri(criteria):
    all_result = {}
    for i in criteria:
        print(i)
        result = generate_onc_cri(i)
        print(result)
        all_result[i] = result
    with open(f'{model}_cat_1_2_decoded_intervened.json', 'w') as f:
        from pprint import pprint
        pprint(all_result)
        json.dump(all_result, f)

c = ["bi_gt_capital", "bi_gt_country", 'abs_icl_country', 'abs_capital',]
add_intervention_answer()
generate_all_cri(c)
