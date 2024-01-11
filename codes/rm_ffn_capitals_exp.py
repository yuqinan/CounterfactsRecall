import json
import sys
sys.path.append('..')
from modeling import load_gptj, GPTJWrapper, load_gpt2xl, load_gpt2, load_pythia, GPT2Wrapper, PythiaWrapper#, #load_bloom, #BloomWrapper, LambdaLayer
from bigbench_tasks import PromptBuilder #load_bigbench_task, multiple_choice_query, PromptBuilder
from rich.progress import track
from console import console, timer
import numpy as np
import torch
import torch.nn as nn
from utils import get_probs_and_mrrs, from_layer_logits_to_prob_distros #model, logits, answer ; logits
import random
import gc
import torch
import pandas as pd
import argparse
import csv

random.seed(42)

class CapitalsPrompter(PromptBuilder):
    def __init__(self, dataset, is_extractive=False, from_memory=False):
        #dataset is a pandas dataframe in this case
        super(CapitalsPrompter, self).__init__(dataset)
        self.is_extractive=is_extractive
        self.from_memory=from_memory
        if from_memory:
            self.is_extractive=False
        #countries = dataset['country'].tolist()
        #capitals  = dataset['']

    def build_mc_prompt(self, index, include_answer=False):
        return None

    def get_n_random_cities(self, n=8):
        cities = self.dataset['capital'].tolist()
        idxs = list(range(len(cities)))
        random.shuffle(idxs)
        return [cities[i] for i in idxs[:n]]

    def build_open_prompt(self, index, include_answer=False):
        """
        nshots: int number of shots. 0 means just return dataset[index] prompt. 1 means 1 example and 1 test prompt
        dataset a bigbench dataset that can be indexed by index. dataset[index]
        index: the index of the test example. This method will use the [index-nshots, index) datapoints as the nshot examples, wrapping if necessary
        returns: a string prompt, the answer as a string
        """
        #country -country capital -capital
        file_path = 'world_capitals.csv'  # Path to the CSV file

        c = []
        with open(file_path, 'r') as file:
            reader = csv.DictReader(file)
            all_title = {}
            for row in reader:
                c.append((row['country'], row['capital']))


        if index == -1:
            index = len(self.dataset['country'].tolist())-1
        for idx, i in enumerate(c):
            if self.dataset['country'][index] in i[0]:
                original_capital = i[1]
            if self.dataset['country'][index] == "Cocos Islands":
                original_capital = "West Island"


        country, gt = self.dataset['country'][index], [self.dataset['capital'][index], original_capital, self.dataset['country'][index]]
        gt_s = self.dataset['capital'][index]
        if False:
            #country, gt = self.dataset['country'][index], [self.dataset['counter'][index], self.dataset['capital'][index], self.dataset['country'][index]]
            country, gt = 'Australia', [self.dataset['counter'][index], 'Canberra', 'Australia']
            gt_s =  self.dataset['counter'][index]
        #inputs = f"""Q: What is the capital of {country}?\nA:"""
        inputs =f"""The capital of {country} is {gt_s}.\nQ: What is the capital of {country}?\nA:"""
        #inputs = datapoint['inputs']#, datapoint['multiple_choice_targets']
        #gt = datapoint['targets'][0].title() #captialize the first letter!
        #gt_idx = targets.index(gt)
        prompt = inputs
        if include_answer:
            prompt+=' '+gt_s
        return prompt, gt_s , [' '+g for g in gt]

    def nshot_open_prompt(self, nshots, index):
        """
        nshots: int number of shots. 0 means just return dataset[index] prompt. 1 means 1 example and 1 test prompt
        dataset a bigbench dataset that can be indexed by index. dataset[index]
        index: the index of the test example. This method will use the [index-nshots, index) datapoints as the nshot examples, wrapping if necessary
        """
        ex_sep = '\n'
        prompt = ''
        cities = []
        countries = []
        for i in range(index-nshots, index):
            p, gt_s, gt = self.build_open_prompt(i, include_answer=True)
            cities.append(gt_s)
            #countries.append(self.dataset['country'].tolist()[i])
            countries.append(p)
            #prompt+=p+ex_sep
            prompt=p


        final_prompt, gt_s, gt = self.build_open_prompt(index)
        cities.append(gt_s) #add the final to the list
        #countries.append(self.dataset['country'][index])
        countries.append(gt_s)
        prompt+=final_prompt
        if True:#self.is_extractive:
            ext_text = ""
            for count, city in zip(countries, cities):
                if self.is_extractive:
                    ext_text+=f"The capital of {count} is {city}.\n"
                elif not self.from_memory:
                    ext_text+=f"The capital of {count} is {city.lower()}.\n"
            #rand_cities = self.get_n_random_cities(10-len(cities))
            #cities+=rand_cities
            #random.shuffle(cities)
            #ext_text = f"Here is a list of cities: {', '.join(cities)}.\n"
            prompt=ext_text+prompt
        console.print()
        console.print(prompt)
        return prompt, gt_s, gt


def generate_ans(model, prompter, idx):
    #making the prompt
    prompt, gt_s, gt = prompter.build_open_prompt(idx) #changed from nshot_open_prompt(self, nshots, index)
    print(gt_s)
    targets = gt#prompter.get_mc_targets(idx)
    gt_idx = targets.index(' '+gt_s)

    #print(prompt, idx, gt)
    #running prompt thru model
    prompt_ids = model.tokenize(prompt)
    logits = model.get_layers(prompt_ids)

    #probs_results = {}
    #rrs_results   = {}
    correct = 0
    gt_correct = 0

    #for i in range(len(targets)):
    
        #probs, rrs = get_probs_and_mrrs(model, logits, tgt)
    decoded = tokenizer.decode(model.model.generate(prompt_ids, max_new_tokens = len(prompt_ids)+15, temperature = 0)[0])
    import re
    if len(re.findall(targets[0],decoded))>1:
        correct = 1
    if targets[1] in decoded:
        if decoded.index(targets[1]) > decoded.index('A:'):
            gt_correct = 1
        #probs_results[i] = probs.tolist()
        #rrs_results[i] = rrs.tolist()
        #if i == gt_idx and rrs[-1] == 1.:
            #correct=1
        #if i == 1 and rrs[-1] == 1.:
            #gt_correct=1


    top10_per_layer = model.topk_per_layer(logits, 10)
    prompt_results = {'inputs':prompt, 'correct':correct, 'gt_correct': gt_correct, 'targets':targets, 'decoded': decoded}#'probs':probs_results, 'rrs':rrs_results, 'top10_per_layer':top10_per_layer}
    from pprint import pprint
    print(prompt_results, flush = True)
    #Lastly, turn the logits into a bunch of prob distributions over the whole vocab. This will be stored separately
    prob_distros = None#from_layer_logits_to_prob_distros(logits)
    return prompt_results, prob_distros, correct

def get_open_generations(model, dataset, is_extractive, from_memory):
    output = []
    num_correct = 0
    prompter = CapitalsPrompter(dataset, is_extractive=is_extractive, from_memory=from_memory)
    #all_prob_distros = []
    with torch.no_grad():
        for i in track(range(len(prompter)), description='iterating...'):
            json_out, prob_distros, correct = generate_ans(model, prompter, i)
            output.append(json_out)
            num_correct+=correct
            #all_prob_distros.append(prob_distros)

    return output, num_correct

    #all_prob_distros = np.stack(all_prob_distros)
    #print(all_prob_distros.shape)
    #np.save(f'{model_name}_{nshots}_open_caps_vocab_distros.npy', all_prob_distros)


def rm_ffn_from_model(model, rm_layers_num):
    b4_params = sum([param.numel() for param in model.parameters() if param.requires_grad])
    layer_start = max(0, len(model.transformer.h)-rm_layers_num)
    console.print(f"REMOVING LAYERS STARTING AT {layer_start}")
    for i in range(layer_start,len(model.transformer.h)):
        model.transformer.h[i].mlp = nn.Identity()
    after_params = sum([param.numel() for param in model.parameters() if param.requires_grad])
    console.print(f"Original # of parameters {b4_params}. After Rm FFN: {after_params}")
    console.print(f"% params removed: {100*((b4_params-after_params)/b4_params)}")
    return model

class BloomIdentityLayer(nn.Module):
    def __init__(self):
        super(BloomIdentityLayer, self).__init__()
    def forward(self, x, y):
        return x+y #bloom expects the MLP to handle the residual connection

def bloom_ffn_from_model(model, rm_layers_num):
    b4_params = sum([param.numel() for param in model.parameters() if param.requires_grad])
    layer_start = max(0, len(model.transformer.h)-rm_layers_num)
    console.print(f"REMOVING LAYERS STARTING AT {layer_start}")
    for i in range(layer_start,len(model.transformer.h)):
        model.transformer.h[i].mlp = BloomIdentityLayer()#nn.Identity()
    after_params = sum([param.numel() for param in model.parameters() if param.requires_grad])
    console.print(f"Original # of parameters {b4_params}. After Rm FFN: {after_params}")
    console.print(f"% params removed: {100*((b4_params-after_params)/b4_params)}")
    return model


def save_output(output, model_name, nshots, is_extractive, removed_layers, counter):
    if is_extractive:
        extabs = 'ext'
    else:
        extabs = 'abs'
    if counter:
        c = "counter"
    else:
        c = "gt"
   # with open(f'{model_name}_{nshots}_{extabs}_open_caps_rm_{removed_layers}_{c}_results_copy.json', 'w') as fp:
    with open(f'{model_name}_decoded.json', 'w') as fp:
        json.dump(output, fp, indent=4)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("model_name", help="name of model to be used")
    parser.add_argument("nshots", type=str, help="number of shots to be used")
    parser.add_argument("--is_extractive", action='store_true', help="whether or not the task is extractive")
    parser.add_argument("--counter", action='store_true', help="whether or not the task is extractive")
    parser.add_argument("--rm_ffn", default=0, type=int, help='how many layers to start removing from the top')
    parser.add_argument("--memory_only", action='store_true', help='pass if you want no context before the Qs')
    args = parser.parse_args()
    model_name = args.model_name#sys.argv[1]
    nshots = args.nshots#sys.argv[2]
    is_extractive = args.is_extractive
    remove_ffn = args.rm_ffn
    memory_only = args.memory_only
    counter = args.counter
    console.print('extractive?', is_extractive, 'counter', counter, 'from memory?', memory_only, 'removing', remove_ffn, 'layers')
    console.print(model_name, nshots, 'shot(s)')
    
    timer_task = timer.add_task("Loading model")
    with timer:
        if 'gpt2' in model_name:#model_name == 'gpt2-xl':
            model, tokenizer = load_gpt2(model_name)
            model = GPT2Wrapper(model, tokenizer)#GPTJWrapper(gptj, tokenizer)
                
        elif 'gptj' == model_name:
            model, tokenizer = load_gptj()
            model = GPTJWrapper(model, tokenizer)
        elif 'bloom' in model_name:
            model, tokenizer = load_bloom(model_name)
            model = BloomWrapper(model, tokenizer)

        elif 'pythia' in model_name:
            model, tokenizer = load_pythia(model_name)
            model = PythiaWrapper(model, tokenizer)
            
    timer.stop_task(timer_task)
    timer.update(timer_task, visible=False)
 
    dataset1 = pd.read_csv('massive_all_combos.csv')
    dataset2 = pd.read_csv("additional_country.csv")
    dataset = pd.concat([dataset1, dataset2], ignore_index=True)
    #get_mc_generations(model, dataset)
    
    num_layers=model.num_layers
    rrange= [0]#list(range(0,num_layers,num_layers//6))
    #rrange.append(num_layers)
    console.print(f"RRANGE: {rrange} .. Num Layers:", num_layers)
    accuracies = []
    for r in rrange:#[12,24,36,48,60,70]:
        if r>0:
            if 'bloom' in model_name:
                model.model = bloom_ffn_from_model(model.model, r)
            else:
                model.model = rm_ffn_from_model(model.model, r)

        if not torch.cuda.is_available():
            model.model = model.model.float()

        if type(nshots)==str and ',' in nshots:
            rnshots = [int(n) for n in nshots.split(',')]
            for n in rnshots:
                console.print("NSHOTS", n)
                nshots = n
                output, num_correct = get_open_generations(model, dataset, is_extractive, memory_only)
                save_output(output, model_name, n, is_extractive, r, counter)
                accuracies.append(num_correct/len(output))
                console.print(accuracies[-1], '%  accuracy')

        else:
            nshots = int(nshots)
            output, num_correct = get_open_generations(model, dataset, is_extractive, memory_only)
            save_output(output, model_name, nshots, is_extractive, r, counter)
            #accuracies.append(num_correct/len(output))
            #console.print(accuracies[-1], '%  accuracy')
    print("Accuracies", accuracies)