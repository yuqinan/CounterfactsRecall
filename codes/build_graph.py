import json
import sys
import numpy as np
from pprint import pprint

over = {}
def load_json(path = "gpt2-large_0_abs_open_caps_rm_0_gt_results_copy.json"):
    filtered = open("output0421/gpt2-large_0_ext_open_caps_rm_0_counter_results_filtered.json")
    filtered =  json.load(filtered)
    gt_correct = [f['targets'][2] for f in filtered if f["gt_correct"]==1]
    correct = [f['targets'][2] for f in filtered if f["correct"]==1]

    f = open(path)
    data= json.load(f)
    rrs_overall = {}
    rrs_overall["0"] = []
    rrs_overall["1"] = []

    rrs_incorrect = {}
    rrs_incorrect["0"] = []
    rrs_incorrect["1"] = []
    if 'counter' in path:
        rrs_overall["2"] = []
        rrs_incorrect["2"] = []
    for d in data:
        if d["targets"][0] in correct:
            rrs_overall["0"].append(d["rrs"]["0"])
            rrs_overall["1"].append(d["rrs"]["1"])
            if 'counter' in path:
                rrs_overall["2"].append(d["rrs"]["2"])
        elif d["targets"][0] in gt_correct:
            rrs_incorrect["0"].append(d["rrs"]["0"])
            rrs_incorrect["1"].append(d["rrs"]["1"])
            if 'counter' in path:
                rrs_incorrect["2"].append(d["rrs"]["2"])
    
    print(len(rrs_overall["0"]))
    print(len(rrs_incorrect["0"]))
    
    rrs_overall["0"] = list(np.mean(np.stack(rrs_overall["0"]), axis = 0))
    rrs_overall["1"] = list(np.mean(np.stack(rrs_overall["1"]), axis = 0))
    if 'counter' in path:
        rrs_overall["2"] = list(np.mean(np.stack(rrs_overall["2"]), axis = 0))
    
    rrs_incorrect["0"] = list(np.mean(np.stack(rrs_incorrect["0"]), axis = 0))
    rrs_incorrect["1"] = list(np.mean(np.stack(rrs_incorrect["1"]), axis = 0))
    
    

    if 'counter' in path:
        rrs_incorrect["2"] = list(np.mean(np.stack(rrs_incorrect["2"]), axis = 0))
    return {"correct": rrs_overall, "incorrect": rrs_incorrect}




#path1 = "gpt2-medium_1_abs_open_caps_rm_0_gt_results.json"
#over["abs_0_gt"] = load_json(path1)
## gpt2-large_0_ext_open_caps_rm_0_counter_results_filtered
path2 = "gpt2-large_0_abs_open_caps_rm_0_gt_results_copy.json"
over["ext_0_counter"] = load_json(path2)
#path3 = "gpt2-medium_1_ext_open_caps_rm_0_counter_results.json"
#over["ext_1_counter"] = load_json(path3)
#path4 = "gpt2-medium_0_ext_open_caps_rm_0_gt_results.json"
#over["ext_0_gt"] = load_json(path4)
#path5 = "gpt2-medium_1_ext_open_caps_rm_0_gt_results.json"
#over["ext_1_gt"] = load_json(path5)

pprint(over)