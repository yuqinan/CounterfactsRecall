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
#parser.add_argument('--scale', help='which set of models')
#parser.add_argument('--head', help='which set of models')
#parser.add_argument('word', help='word to search')
#
args = parser.parse_args()

model = args.model
#scale = args.scale
#head = args.head


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
            sublists.append(list(set(sublist)))
        if criteria == "bi_gt_country":
            sublist = [x['capital'] for x in input if percentiles[i] <= float(x[criteria]) <= percentiles[i + 1]]
            sublists.append(list(set(sublist)))
        
        if criteria == 'abs_icl_country':
            sublist = [x['country'] for x in input if percentiles[i] <= float(x[criteria]) <= percentiles[i + 1]]
            sublists.append(list(set(sublist)))
        if criteria == 'bi_gt_capital':
            sublist = [x['country'] for x in input if percentiles[i] <= float(x[criteria]) <= percentiles[i + 1]]
            sublists.append(list(set(sublist)))
        
        if criteria == 'bi_pairs_icl':
            sublist = [[x['capital'], x['country']] for x in input if percentiles[i] <= float(x[criteria]) <= percentiles[i + 1]]
       
            sublists.append(sublist)

    for i, sublist in enumerate(sublists):
        print(f'---- sublist {i} -------')
        print(sublist)

    return sublists

def generate_occ_for_one_model(name, criteria):

    data = read_csv_as_dict(f"{name}_big_results_decoded.csv")
    occ = [float(d[criteria]) for d in data]

    return occ, data


def generate_samples(criteria, path = f"final_{model}_camera_ready.json"):

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
                        data['rank'] = rank
                        ranked_result[rank].append(data)
                if 'abs_icl_country' == criteria or 'bi_gt_capital' == criteria:
                    if c in cp:
                        data['rank'] = rank
                        ranked_result[rank].append(data)
                
                if 'bi_pairs_icl' == criteria:
                    if [data['targets'][0][1:].lower(), c] in cp:
                        data['rank'] = rank
                        ranked_result[rank].append(data)                       
    
    with open(f'part3/{model}/{model}_{criteria}_decoded_with_answers_camera_ready.json', 'w') as f:
        json.dump(ranked_result, f)


def generate_onc_cri(criteria, s):

    generate_samples(criteria)

    with open(f'part3/{model}/{model}_{criteria}_decoded_with_answers_camera_ready.json', 'r') as f:
        data = json.load(f)
    
    result = {'gt': [], 'ct': [], f'i_gt': [], f'i_ct':[]}
    
    for k, v in data.items():
        result['gt'].append(len([i for i in v if i['gt_correct'] == 1])/len(v))
        result['ct'].append(len([i for i in v if i['correct'] == 1])/len(v))
        result[f'i_gt'].append(len([i for i in v if i[f'intervened_gt_correct_{s}'] == 1])/len(v))
        result[f'i_ct'].append(len([i for i in v if i[f'intervened_correct_{s}'] == 1])/len(v))
                
    return result

def generate_all_cri(criteria, s):

    all_result = {}
    for i in criteria:
        print(i)
        result = generate_onc_cri(i, s)
        print(result)
        all_result[i] = result
    with open(f'part3/{model}/{model}_{s}_criteria_camera_ready.json', 'w') as f:
        from pprint import pprint
        pprint(all_result)
        json.dump(all_result, f)

if model == 'pythia-1.4b':
    scale = [-1.5, 10.0, 1.5, -6.0]
if model == 'pythia-2.8b':
    scale = [-2.8, 10.0, 2.6, -1.5]
if model == 'gpt2-xl':
    scale = ['-6.8_capital', '-10.0_capital', '10.0_capital_memory','10.0_capital_icl']
    
c = ["bi_gt_capital", "bi_gt_country", 'abs_icl_country', 'abs_capital']
for s in scale:
    generate_all_cri(c, s)






    
