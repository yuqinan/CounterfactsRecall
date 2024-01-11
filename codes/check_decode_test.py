import json
from pprint import pprint
import random

def check_answer(d):
    input = d['inputs']
    decoded_input = d['decoded']
    generated_answer_all = decoded_input[len(input):]
    if "Q:" not in generated_answer_all:
        generated_answer = generated_answer_all
    else:
        generated_answer = generated_answer_all[:generated_answer_all.index("Q:")]
    return d['targets'][2] in generated_answer
#point3

with open('final_pythia-1.4b.json', 'r') as f:
    data= json.load(f)
    print(len(data))
    none = [i for i in data if i['intervened_gt_correct_-0.7'] ==0 and i['intervened_correct_-0.7']==0 and check_answer(i) == 0]
    print(len(none))  

    #none = [i for i in data if i['correct']==0 and i['gt_correct']==1 and i['intervened_correct_-2.8'] == 1]
    ##pprint(none[2000:2300])
#point2
#with open('final_pythia-1.4b.json', 'r') as f:
#    data= json.load(f)
#    none = [i for i in data if i['correct']==0 and i['gt_correct'] == 0]
#    pprint(len(none))
#    none = [i for i in none if check_answer(i)]
#    pprint(len(none))
#    none = [i for i in none if "A: The capital" in i['intervene_decoded_10.0'] and i['intervened_correct_10.0']==1]
#    pprint(len(none))
#    #none = [i for i in data if i['correct']==0 and i['gt_correct']==1 and i['intervened_correct_-2.8'] == 1]
#    ##pprint(none[2000:2300])
#
#point 1
"""     none = [i for i in data if i['correct']==1 and ]
    pprint(len(none))
    none = [i for i in data if "A: The capital" in i['decoded']]
    pprint(len(none))
    none = [i for i in data if "A: The capital" in i['decoded'] and i['correct']==1]
    pprint(len(none))
    none = [i for i in data if "A: The capital" in i['intervene_decoded_-0.7']]
    pprint(len(none))
    none = [i for i in data if "A: The capital" in i['intervene_decoded_-0.7'] and i['intervened_correct_-0.7']==1]
    pprint(len(none)) """
