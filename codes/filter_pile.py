from transformers import AutoTokenizer
import json
import re
from tqdm import tqdm
import zstandard as zstd
from functools import partial
from pprint import pprint
import csv

import argparse

# Create an ArgumentParser object
parser = argparse.ArgumentParser()
#
## Define command-line arguments
parser.add_argument('--subset', help='word to search')
#parser.add_argument('word', help='word to search')
#
args = parser.parse_args()

subset = args.subset

def get_capital():
    file_path = 'world_capitals.csv'  # Path to the CSV file
    word = []
    with open(file_path, 'r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            word.append(row['capital'].lower())
    return word

def get_country():
    file_path = 'world_capitals.csv'  # Path to the CSV file
    word = []
    with open(file_path, 'r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            if row['country'].lower() == "cocos (keeling) islands":
                word.append("cocos islands")
            if row['country'].lower() == "republic of china (taiwan)":
                word.append("taiwan")
                word.append("republic of china")
            if row['country'].lower() == 'east timor (timor-leste)':
                word.append("east timor")
                word.append("timor-leste")
            if row['country'].lower() == 'united kingdom; england':
                word.append("united kingdom")
                word.append("england")
            word.append(row['country'].lower())
           
    return word


capital, country = get_capital(), get_country()

a_input = capital+country

import numpy as np

def search_cooccurence(seq1, window_size = 2048, capital=capital, country=country, tokenizer = AutoTokenizer.from_pretrained(f'EleutherAI/pythia-1b')):
    
    basic_count = {}

    for co in a_input:
        basic_count[co] = 0

    count = {}
    for co in country:
        count[co] = {}
        for ca in capital:
            count[co][ca] = 0
            
    with open(f'subset_{subset}/{seq1}.json', 'r') as file:
        filtered = json.load(file) ## this is a list of key
    
    key = 0
    with zstd.open(open(f'/gpfs/data/epavlick/datasets/pile/train/{seq1}.jsonl.zst', "rb"), "rt", encoding="utf-8") as f:
        for row in f:
            key += 1
            if not(key > int(subset) * 3500000 and key < (int(subset) + 1) * 3500000 + 1):
                continue
            print(key, flush = True)
            data = json.loads(row)
            pattern = r'\b|\b'.join(a_input)
            pattern = r'\b'+ pattern + r'\b'
            se = re.finditer(pattern, data['text'], re.IGNORECASE)
            a = [(match.group(0).lower(), match.start()) for match in se]               
            appearance = [i[0] for i in a]
            #index = [i[1] for i in a]
            l = list(set(appearance))
            #in_capital = [i for i in l if i in capital]
            #in_country = [i for i in l if i in country]
            #l = [i for i in l if i is not ' washington, d.c, ']
            ######### This is to calculate occurences ##########
            for ap in l:
                try:
                    basic_count[ap] += len([i for i in appearance if i == ap])
                except KeyError:
                    basic_count[ap] = len([i for i in appearance if i == ap])
                    print(seq1, ap, key, basic_count)

                
                
                ######### This is to calculate coocurences
                #little_dictionary_country = {}
                #little_dictionary_capital = {}
                #for co in in_country:
                #    little_dictionary_country[co] = []
                #for ca in in_capital:
                #    little_dictionary_capital[ca] = []
        #
                #for idx, i in enumerate(appearance):
                #    if i in in_country:
                #        little_dictionary_country[i].append(index[idx])
                #    if i in in_capital:
                #        little_dictionary_capital[i].append(index[idx])
                #
             #
                #if len(in_capital) == 0 or len(in_country) == 0:
                #    continue

#                tokens = tokenizer.tokenize(data['text'])
#                if len(tokens) <= window_size:
#                    for in_co in in_country:
#                        for in_ca in in_capital:
#                            count[in_co][in_ca] += 1
#
#                     
#                else:
                #for key_country, value_country in little_dictionary_country.items():
                #    for key_capital, value_capital in little_dictionary_capital.items():
                #        from itertools import product
                #        differences = [abs(x - y) for x, y in product(value_country, value_capital)]
                #        diff = len([i for i in differences if i < 4 *window_size])
                #        count[key_country][key_capital] += diff
            
            if key % 1000 == 0:
                #with open(f"c_more_{subset}/{seq1}.json", 'w') as f:
                #        json.dump(count, f)
                with open(f"/gpfs/data/epavlick/overwrite/exact_count_{subset}/{seq1}.json", 'w') as f2:
                        json.dump(basic_count, f2)
                                

#search_cooccurence('12')
import numpy as np

seq = [
 '00',
 '01',
 '02',
 '03',
 '04',
 '05',
 '06',
 '07',
 '08',
 '09',
 '10',
 '11',
 '12',
 '13',
 '14',
 '15',
 '16',
 '17',
 '18',
 '19',
 '20',
 '21',
 '22',
 '23',
 '24',
 '25',
 '26',
 '27',
 '28',
 '29']

#search_cooccurence('01')


from multiprocessing import Pool
    
def process_files(seq):
    # Create a multiprocessing pool with the number of desired processes
    pool = Pool(processes=len(seq))
    # Map the process_file function to each file path in parallel
    with Pool() as pool:
        # Use starmap to pass multiple arguments to the process_file function
        pool.map(search_cooccurence, seq)

process_files(seq)


#################  Generate Coocurences ##################
def get_co(seq):

    count = {}
    for co in country:
        count[co] = {}
        for ca in capital:
            count[co][ca] = 0 
    
    for i in seq:
        with open(f'cooccurences_1/{i}.json', 'r') as f:
            data = json.load(f)
            for k, v in data.items():
                for ik, iv in v.items():
                    count[k][ik] += iv
        with open(f'cooccurences_0/{i}.json', 'r') as f:
            data = json.load(f)
            for k, v in data.items():
                for ik, iv in v.items():
                    count[k][ik] += iv

    with open("coocurences.json", 'w') as f:
        json.dump(count, f)

################### COUNT RATIO #############################
def count_ratio(seq, tokenizer = AutoTokenizer.from_pretrained(f'EleutherAI/pythia-1b')):
    all_ratio = []
    with zstd.open(open(f'/gpfs/data/epavlick/datasets/pile/train/{seq}.jsonl.zst', "rb"), "rt", encoding="utf-8") as f:
        key = 0
        for row in f:
            
            key = key + 1
            data = json.loads(row)
            chara = len(data['text'])
            tokens = len(tokenizer.tokenize(data['text']))
            all_ratio.append(tokens/chara)
            if key % 10000 == 0:
                print(key)
                try:
                    with open(f'count_ratio.json', 'r') as f2:
                        d = json.load(f2)
                        d.extend(all_ratio)
                        all_ratio = []
                    with open(f'count_ratio.json', 'w') as f2:
                        json.dump(d, f2)
                except FileNotFoundError:
                    with open(f'count_ratio.json', 'w') as f2:
                        json.dump(all_ratio, f2)
                        all_ratio = []
