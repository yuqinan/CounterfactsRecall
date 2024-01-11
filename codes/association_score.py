import json
import pandas as pd
import numpy as np
def load_dataset():

    overall = []
    with open('pythia-1b_capital_country.json') as f:
        given_country_ask_capital = json.load(f)
   
    with open('pythia-1b_country_capital') as f2:
        given_capital_ask_country = json.load(f2)  

    for idx,d in enumerate(given_capital_ask_country):
        current = {}
        current["country"] = d["targets"][0]
        current["capital"] = ' '+d['inputs'][:d["inputs"].index(' is')]
        current["given_capital_ask_country"] = d['rrs']['0'][-1]
        current["given_country_ask_capital"] = [i for i in given_country_ask_capital if i['targets'][0][1:] == d['inputs'][:(d["inputs"].index(' is'))]][0]['rrs']['0'][-1]
        overall.append(current)
    
    return overall

import csv

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
    

#overall = load_dataset()
#write_dicts_to_csv(overall, "pythia_association_scores.csv")

def load_csv_as_dicts(filename):
    dicts = []
    with open(filename, 'r') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            dicts.append(dict(row))
    return dicts

def get_confidence_score():

    with open('pythia-1b_association_output.json') as f:
        all_data = json.load(f)
    
    def filtered_country(country):
        all_scores = []
        for i in all_data:
            if i['inputs'][i['inputs'].index(' of ') + 3:i['inputs'].index(' is ')] == country:
                all_scores.append(i['rrs']['0'][-1])
        if len(all_scores) is not 247:
            print(country)
        return np.mean(all_scores)
        
    def filtered_counter_capital(capital):
        all_scores = []
        for i in all_data:
            if i['targets'][0] == capital:
                all_scores.append(i['rrs']['0'][-1])
        if len(all_scores) is not 247:
            print(capital)
        return np.mean(all_scores)


    all_pairs = load_csv_as_dicts('pythia_association_scores.csv')
    for idx, data in enumerate(all_pairs):
        data["overwrite_rrs_country"]= filtered_country(data['country'])
        data["overwrite_rrs_capital"]= filtered_counter_capital(data['capital'])
        all_pairs[idx] = data

    write_dicts_to_csv(all_pairs, "pythia-1b_association_scores_with_confidence.csv")
    


get_confidence_score()

def generate_all_combos(path = "world_capitals.csv"):
    world_capital = load_csv_as_dicts(path)
    country = [i['country'] for i in world_capital]
    capital = [i['capital'] for i in world_capital]
    all_combos = []
    for idi, i in enumerate(country):
        for idj, j in enumerate(capital):
            if idi is not idj:
                all_combos.append({'country':i, 'capital':j})
    
    write_dicts_to_csv(all_combos, "massive_all_combos.csv")
                



#generate_all_combos()