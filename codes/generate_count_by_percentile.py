
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
parser.add_argument('model', help='word to search')
parser.add_argument('type', help='word to search')
#parser.add_argument('word', help='word to search')
#
args = parser.parse_args()

model = args.model
criteria = args.type

def write_dicts_to_csv(dicts, filename):
    # Extract the keys from the first dictionary to use as fieldnames
    fieldnames = dicts[0].keys()

    with open(filename, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        # Write the header row
        writer.writeheader()

        # Write the dictionary values as rows
        for dictionary in dicts:
            writer.writerow(dictionary) 

### capital overwrite ##
def filter_capital(model):
    file_path = 'world_capitals.csv'  # Path to the CSV file
    all_capital = {}


    with open(file_path, 'r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            all_capital[row['capital']] = {'capital':row['capital'], 'ct':0, 'gt':0}

    filtered = open(f"{model}_association_output_new.json")
    filtered =  json.load(filtered)
    for f in filtered:
        if f["gt_correct"]==1:
            all_capital[f['targets'][0][1:]]['gt'] += 1
        if f["correct"]==1:
            all_capital[f['targets'][0][1:]]['ct'] += 1
    all_capital_list = []
    for key, i in all_capital.items():
        all_capital_list.append(i)
    write_dicts_to_csv(all_capital_list, f"{model}_overwrite_sum_by_capital.csv")
   



def build_graph(path="gpt2-large_overwrite_sum_by_capital.csv"):
    if criteria == "abs":
        with open("current_counts.json", 'r') as json_file:
            count = json.load(json_file)
    else:
        with open("city_capital_occ.json", 'r') as json_file:
            count = json.load(json_file)
    
    with open(path, 'r') as f3:
        reader = csv.DictReader(f3)
        all_capital = []

        for row in reader:
            if row['capital'] == 'Nicosia' or row['capital'] == 'Kingston' or row['capital'] == 'Jerusalem':
                continue
            try:
                all_capital.append({"capital": row['capital'], 
                                    "ct": row['ct'], 
                                    'gt': row['gt'], 
                                    "occ": count[row['capital'].lower()]})
            except KeyError:
                continue
    
    
    ct = [i['ct'] for i in all_capital]
    gt = [i['gt'] for i in all_capital]
    occ = [i['occ'] for i in all_capital]
    
    return {'ct': ct, 'gt': gt, 'occ': occ}, all_capital

def p(lsort, input):   
    import numpy as np

    # Calculate percentiles
    percentiles = np.percentile(lsort, range(0, 101, 10))

    # Break the list into sublists
    sublists = []
    for i in range(len(percentiles) - 1):
        sublist = [x for x in input if percentiles[i] <= x['occ'] <= percentiles[i + 1]]
        sublists.append(sublist)

    for i, sublist in enumerate(sublists):
        print(f'---- sublist {i} -------')
        for s in sublist:
            print(s)

    return sublists


def finalize():
    if model == 'gpt':
        model_series = ['gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl']

    if model == 'pythia':
        model_series = ['pythia-70m','pythia-160m', 'pythia-410m', 'pythia-1b', 'pythia-1.4b', 'pythia-2.8b']

    all_output = {}

    for i_a in model_series:

        filter_capital(i_a)

        lsort, input = build_graph(f'{i_a}_overwrite_sum_by_capital.csv')
        lsort = lsort['occ']
        print(lsort)
        sublists = p(lsort, input)
        ct = [np.mean([int(i['ct']) for i in j]) for j in sublists]
        gt = [np.mean([int(i['gt']) for i in j]) for j in sublists]


        with open(f"{i_a}_percentile.json","w") as f:
            all_output[i_a] = {'ct': ct, 'gt':gt}
            json.dump({'ct': ct, 'gt':gt}, f)
        
    with open(f"all_{model}_percentile_{criteria}.json","w") as f:
        json.dump(all_output, f)


finalize()
