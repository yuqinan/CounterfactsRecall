import json
import sys
sys.path.append('..')
from rich.progress import track
from console import console, timer
import numpy as np
import random
import pandas as pd
import csv

### There are two conditions we want to make sure. The model knows where the capital of country is. And which country the **counter** capital belongs to
# Load the JSON data into a Python object
with open('gpt2-large_0_ext_open_caps_rm_0_counter_results_Australia.json') as f:
    data = json.load(f)

all_counters_in_confidenct = [d["targets"][0] for d in data]

all_tuples = [d["targets"][0] for d in data if d["correct"] ==1 ] ## all the tuples that stores country <-> captial relations

dataset = pd.read_csv("contain_gt.csv") #all this data sticks to ground truth 
dataset = dataset.to_dict(orient='records')
correct = [d["capital"] for d in dataset]

filtered = []
count = 0

for d in dataset:
    if d["counter"] in correct:
        count = count + 1
    #id_x = all_counters_in_confidenct.index(" "+d["counter"])
    #if ([" "+d["capital"], " "+d["country"]] in all_tuples) and ([data[id_x]['targets'][0], data[id_x]['targets'][1]] in all_tuples):
    if " "+d["counter"] in all_tuples and d["counter"] in correct:
        filtered.append(d)
print(count)
print(len(filtered))
# Open a new CSV file
#with open('Australia_ct.csv', 'w', newline='') as f:
#    writer = csv.writer(f)
#    writer.writerow(filtered[0].keys())
#    for row in filtered:
#        writer.writerow(row.values())