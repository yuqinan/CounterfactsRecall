import json
import sys
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("model_name", help="name of model to be used")
args = parser.parse_args()
model_name = args.model_name

def generate_include():
    with open(f'{model_name}_decoded_without_counters.json', 'r') as f:
        original = json.load(f)
        clean = []
        for idx, i in enumerate(original):
            if i['gt'] in i['decoded']:
                clean.append(i['input_para'])
    return clean
generate_include()