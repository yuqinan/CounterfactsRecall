import argparse
import csv
import re
import pandas as pd
import json
import numpy as np
from tqdm import tqdm
from rm_ffn_capitals_exp import get_open_generations
from modeling import load_gptj, GPTJWrapper, load_gpt2xl, load_gpt2, load_pythia, GPT2Wrapper, PythiaWrapper#, #load_bloom, #BloomWrapper, LambdaLayer
from bigbench_tasks import PromptBuilder #load_bigbench_task, multiple_choice_query, PromptBuilder
from rich.progress import track
from pprint import pprint


def get_capital():
    file_path = 'world_capitals.csv'  # Path to the CSV file
    word = []
    with open(file_path, 'r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            word.append(' '+row['capital'].lower()+' ')
    return word

def get_country():
    file_path = 'world_capitals.csv'  # Path to the CSV file
    word = []
    with open(file_path, 'r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            if row['country'].lower() == "cocos (keeling) islands":
                word.append(" cocos islands ")
            if row['country'].lower() == "republic of china (taiwan)":
                word.append(" taiwan ")
                word.append(" republic of china ")
            if row['country'].lower() == 'east timor (timor-leste)':
                word.append(" east timor ")
                word.append(" timor-leste ")
            if row['country'].lower() == 'united kingdom; england':
                word.append(" united kingdom ")
                word.append(" england ")
            word.append(' '+row['country'].lower()+' ')

    return word

def get_capital_upper():
    file_path = 'world_capitals.csv'  # Path to the CSV file
    word = []
    with open(file_path, 'r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            word.append(' ' +row['capital']+' ')
    return word


capital_l =  get_capital()
country_l = get_country()
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

#
#def create_new_additional_dataset():
#
#    country = ["Cocos Islands", "Taiwan", "Republic of China", "East Timor", "Timor-Leste", "United Kingdom", "England"]
#    
#    addi = []
#    for i in capital:
#        for j in country:
#            addi.append({"country": j, "capital": i})
#    
#    write_dicts_to_csv(addi, "additional_country.csv")
#

#create_new_additional_dataset()


######################## Actual Codes Start here #################
def make_result(model):
    with open('total_pair_count_new.json', 'r') as f:
        occ = json.load(f)

    with open(f'{model}_association_output_new.json', 'r') as f:
        model_result = json.load(f)

    with open(f'total_absolute_count_new.json', 'r') as f:
        count = json.load(f)

    all_result = []
    for result in model_result:
        capital = ' '+result['targets'][0][1:].lower()
        country =  ' '+result['inputs'][result['inputs'].index(' of ')+4:result['inputs'].index(' is ')].lower()
        gt_country = country_l[capital_l.index(capital + ' ')].lower()

        if capital == ' canberra':
            continue

        all_result.append({"country": country[1:], 
            "capital": capital[1:], 
            "gt": result["gt_correct"], 
            'ct': result["correct"], 
            'occ_absolute_capital': count[capital[1:]],
            'occ_pair_capital': occ[gt_country][capital+' '],
            'occ_pair_exact': occ[country + ' '][capital+' '],
            'mi': occ[country + ' '][capital+' '] / count[capital[1:]]})

    write_dicts_to_csv(all_result, f"{model}_big_results.csv")

parser = argparse.ArgumentParser()
#
## Define command-line arguments
parser.add_argument('model', help='word to search')
#parser.add_argument('word', help='word to search')
#
args = parser.parse_args()

model = args.model


import csv

def read_csv_as_dict(filename):
    data = []
    with open(filename, 'r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            data.append(dict(row))
    return data


def p(lsort, input):   
    import numpy as np

    # Calculate percentiles
    percentiles = np.percentile(lsort, range(0, 101, 10))

    # Break the list into sublists
    sublists = []
    for i in range(len(percentiles) - 1):
        sublist = [x for x in input if percentiles[i] <= float(x['mi']) <= percentiles[i + 1]]
        sublists.append(sublist)

    for i, sublist in enumerate(sublists):
        print(f'---- sublist {i} -------')
        pprint(sublist[-20:])

    return sublists

def generate_occ_for_one_model(name):

    data = read_csv_as_dict(f"{name}_big_results.csv")

    occ = [float(d['mi']) for d in data]
    return occ, data



def finalize():

    if model == 'gpt':
        model_series = ['gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl']

    if model == 'pythia':
        model_series = ['pythia-70m','pythia-160m', 'pythia-410m', 'pythia-1b', 'pythia-1.4b', 'pythia-2.8b']

    all_output = {}

    for i_a in model_series:

        make_result(i_a)
        lsort, input = generate_occ_for_one_model(i_a)
        sublists = p(lsort, input)
        ct = [int(np.sum([int(i['ct']) for i in j])) for j in sublists]
        gt = [int(np.sum([int(i['gt']) for i in j])) for j in sublists]


        all_output[i_a] = {'ct': ct, 'gt':gt}
        print(all_output)
        
  


    with open(f"all_{model}_percentile_cross_all.json","w") as f:
        json.dump(all_output, f)

finalize()