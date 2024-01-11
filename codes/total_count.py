import csv
import re
import pandas as pd
import json
import numpy as np
from tqdm import tqdm
from pprint import pprint

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

def build_count_dic():
    file_path = 'world_capitals.csv'  # Path to the CSV file

    with open(file_path, 'r') as file:
        reader = csv.DictReader(file)
        all_title = {}
        for row in reader:
            c = row["country"]
            ca = row["capital"].lower()
            all_title[f'{ca}']= []

    import os

    folder_path = "count"  # Specify the folder path

    # Get a list of all files in the folder
    file_list = os.listdir(folder_path)

    # Iterate over each file
    for file_name in file_list:
        with open(f'count/{file_name}', 'r') as json_file:
            json_data = json.load(json_file)
        keys = json_data.keys()
        file_name_new = file_name[:file_name.index('.json')]
        capital = file_name_new[file_name_new.rfind("-")+1:file_name_new.rfind(".")]
        if "Benin-porto-novo" in file_name:
            capital = 'porto-novo'
        if "Haiti-port-au-prince" in file_name:
            capital = "port-au-prince"
        if "Wallis and Futuna-mata-utu" in file_name:
            capital = "mata-utu"
        if '7000000' in keys:
            all_title[capital].append(json_data['7000000'])


    new_all_title = {}
    for key, value in all_title.items():
        if len(value) != 0:     
            new_all_title[key] = sum(value)
    # Dump the dictionary into a JSON file
    print(len(new_all_title))
    with open("current_counts.json", 'w') as json_file:
        json.dump(new_all_title, json_file)

####### additional ########

    with open('pythia-1b_association_scores_with_confidence.csv', 'r') as file:
        reader = csv.DictReader(file)
        all_capital = []
        for row in reader:
            if row['given_country_ask_capital'] == '1.0' and row['given_capital_ask_country'] == '1.0':
                all_capital.append(row['capital'][1:].lower())
    
    filter_exist = {}
    for key, value in new_all_title.items():
        if key in all_capital:
            filter_exist[key]= value

    capitals = list(filter_exist.keys())
    count = list(filter_exist.values())

    max_count = np.max(count)
    min_count = np.min(count)
    max_capital = capitals[count.index(max_count)]
    min_capital = capitals[count.index(min_count)]
    print(f'max occurence capital {max_capital}, min occurence capital {min_capital}')

#build_count_dic()

### capital overwrite ##
def filter_capital(capital):
    filtered = open("gpt2-large_association_output_new.json")
    filtered =  json.load(filtered)
    gt_correct = [f['targets'][0] for f in filtered if f["gt_correct"]==1 and f['targets'][0] == f' {capital}']
    correct = [f['targets'][0] for f in filtered if f["correct"]==1 and f['targets'][0] == f' {capital}']
    
    can_it_overwrite = {'country': capital, 'ovewrite': len(correct), 'gt_correct': len(gt_correct)}
    print(can_it_overwrite)
    return can_it_overwrite

#all_capital = []
#file_path = 'world_capitals.csv'  # Path to the CSV file
#all_overwrite = []
#with open(file_path, 'r') as file:
#    reader = csv.DictReader(file)
#    all_title = {}
#    for row in reader:
#        all_capital.append(row['capital'])
#
#for i in all_capital:
#    print(i)
#    all_overwrite.append(filter_capital(i))
#
#write_dicts_to_csv(all_overwrite, "gpt2-large_overwrite_sum_by_capital.csv")


#### country overwrite 
def filter_count(country):

    filtered = open("pythia-1b_association_output_new.json")
    filtered =  json.load(filtered)
    can_it_overwrite = []

    gt_correct = [f['targets'][0] for f in filtered if f["gt_correct"]==1 and f['country'][1:] == country]
    correct = [f['targets'][0] for f in filtered if f["correct"]==1 and f['country'][1:] == country]
    
    can_it_overwrite = {'country': country, 'ovewrite': len(correct), 'gt_correct': len(gt_correct)}

    pile_count = open("current_counts.json")
    pile_count = json.load(pile_count)
    
    result = {'correct': [] ,
              'gt_correct': []}
    correct = [i[1:].lower() for i in correct]
    gt_correct = [i[1:].lower() for i in gt_correct]

    for key,value in pile_count.items():
        if key in correct:
            result['correct'].append(value)
        elif key in gt_correct:
            result['gt_correct'].append(value)

    c = len(result['correct'])
    if c!=0:
        result['correct'] = np.mean(result['correct'])
    if c == 0:
        print("c", country)
        result['correct'] = 0
    g = len(result['gt_correct'])
    if g!=0:
        result['gt_correct'] = np.mean(result['gt_correct'])

    if g == 0:
        print("g", country)
        result['gt_correct'] = 0
    return can_it_overwrite, result


#with open("pythia_edited.json") as f:
#    data = json.load(f)
#
#print(len([i for i in data if i['gt_correct'] == 1])/len(data))

#all_country = []
#file_path = 'world_capitals.csv'  # Path to the CSV file
#
#all_result = {'correct': [], 'gt_correct': []}
#all_overwrite = []
#with open(file_path, 'r') as file:
#    reader = csv.DictReader(file)
#    all_title = {}
#    for row in reader:
#        all_country.append(row['country'])
#
#for country in tqdm(all_country):
#    o, a = filter_count(country)
#    if a['correct'] != 0:
#        all_result['correct'].append(a['correct'])
#    if a['gt_correct']!= 0:
#        all_result['gt_correct'].append(a['gt_correct'])
#    all_overwrite.append(o)
#
#all_result['correct'] = np.mean(all_result['correct'])
#all_result['gt_correct'] = np.mean(all_result['gt_correct'])
#
#print(all_result)


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

#write_dicts_to_csv(all_overwrite, "pythia_overwrite_possibility.csv")

def check_gt():
    filtered = open("pythia-1b_association_output.json")
    filtered =  json.load(filtered)

    file_path = 'world_capitals.csv'  # Path to the CSV file
    c = []
    with open(file_path, 'r') as file:
        reader = csv.DictReader(file)
        all_title = {}
        for row in reader:
            c.append((row['country'], row['capital']))

    for idx, f in enumerate(filtered):
        country = f['inputs'][f['inputs'].index(' of ')+3:f['inputs'].index(' is ')]
        for idx, i in enumerate(c):
            if f' {i[0]}' == country:
                original_capital = f' {i[1]}'
        f['targets'].append(original_capital)
        f['country'] = country
        if f["top10_per_layer"][-1][0] in original_capital:
            if original_capital.index(f["top10_per_layer"][-1][0]) == 0:
                f['gt_correct'] = 1
                print("here")

        filtered[idx] = f
    
    with open("pythia_edited.json", 'w') as json_file:
        json.dump(filtered, json_file)

def add_country():
    filtered = open("pythia-1b_association_output_new.json")
    filtered =  json.load(filtered)

    for idx, f in enumerate(filtered):
        country = f['inputs'][f['inputs'].index(' of ')+3:f['inputs'].index(' is ')]
        f['country'] = country
        filtered[idx] = f
    
    with open("pythia-1b_association_output_new.json", 'w') as json_file:
        json.dump(filtered, json_file)
#check_gt()

#with open('output/gpt2-large_0_ext_open_caps_rm_0_counter_results.json') as f:
#    data = json.load(f)
#
#print(len([i for i in data if i['rrs']['0'][-1] == 1])/len(data))
## {'correct': 278012.0172413793, 'gt_correct': 486716.6666666667}

#with open('pythia-1b_association_output_new.json') as f:
#    data = json.load(f)
#
#print(len([i for i in data if i['gt_correct'] == 1])/len(data))
# {'correct': 278012.0172413793, 'gt_correct': 486716.6666666667}
#add_country

#{'correct': 328085.6395320095, 'gt_correct': 516854.4922965081}

#import zstandard as zstd
#from pprint import pprint
#with zstd.open(open(f'/gpfs/data/epavlick/datasets/pile/train/00.jsonl.zst', "rb"), "rt", encoding="utf-8") as f:
#    for row in f:
#        pprint(row)
        


def build_graph(path="gpt2-large_overwrite_sum_by_capital.csv"):

    with open("current_counts.json", 'r') as json_file:
        count = json.load(json_file)
    
    with open(path, 'r') as f3:
        reader = csv.DictReader(f3)
        all_capital = []

        for row in reader:
            if row['country'] == 'Nicosia' or row['country'] == 'Kingston' or row['country'] == 'Jerusalem':
                continue
            try:
                all_capital.append({"capital": row['country'], 
                                    "ct": row['ovewrite'], 
                                    'gt': row['gt_correct'], 
                                    "occ": count[row['country'].lower()]})
            except KeyError:
                continue
    
    
    ct = [i['ct'] for i in all_capital]
    gt = [i['gt'] for i in all_capital]
    occ = [i['occ'] for i in all_capital]
    
    return {'ct': ct, 'gt': gt, 'occ': occ}, all_capital

def percentile(lsort, input):   
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

#lsort, input = build_graph()
#lsort = lsort['occ']
#sublists = percentile(lsort, input)
#ct = [np.mean([int(i['ct']) for i in j]) for j in sublists]
#gt = [np.mean([int(i['gt']) for i in j]) for j in sublists]
#print(ct, gt)

build_count_dic()
