import json
import sys
sys.path.append('..')
from rich.progress import track
from console import console, timer
import numpy as np
import random
import pandas as pd
import csv


def get_confidenct_score(path = "get_counter_filtered.csv"):
    to_check = pd.read_csv(path)
    with open('output0421/gpt2-large_1_abs_open_caps_rm_0_gt_results.json') as f:
        gt_confidence = json.load(f)

    prob = []
    to_check = to_check.to_dict(orient='records')
    all_counters_in_confidenct = [d["targets"][0] for d in gt_confidence]
    for idx, row in enumerate(to_check):
        id_x = all_counters_in_confidenct.index(" "+row["counter"])
        prob_cur = np.mean(gt_confidence[id_x]["probs"]["0"][-5:])
        to_check[idx]["confidence"] = prob_cur
        prob.append(prob_cur)
    
    with open(path, 'w', newline='') as f:
     #Create a CSV writer
        writer = csv.writer(f)

     #Write the header row
        writer.writerow(to_check[0].keys())

     #Write the data rows
        for row in to_check:
            writer.writerow(row.values())
    


    print(np.mean(prob))

#get_confidenct_score("get_counter_filtered.csv")
get_confidenct_score("with_counter.csv")
    
