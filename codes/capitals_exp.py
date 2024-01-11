import json
import sys
sys.path.append('..')
from modeling import load_gptj, GPTJWrapper, load_gpt2xl, load_gpt2, GPT2Wrapper, load_bloom, BloomWrapper, LambdaLayer
import torch.nn as nn
from bigbench_tasks import load_bigbench_task, multiple_choice_query, PromptBuilder
from rich.progress import track
from console import console, timer
import numpy as np
from utils import get_probs_and_mrrs, from_layer_logits_to_prob_distros #model, logits, answer ; logits
import random
import gc
import torch
import pandas as pd

random.seed(42)

class CapitalsPrompter(PromptBuilder):
    def __init__(self, dataset):
        #dataset is a pandas dataframe in this case
        super(CapitalsPrompter, self).__init__(dataset)
        #countries = dataset['country'].tolist()
        #capitals  = dataset['']

    def build_mc_prompt(self, index, include_answer=False):
        return None

    def build_open_prompt(self, index, include_answer=False):
        """
        nshots: int number of shots. 0 means just return dataset[index] prompt. 1 means 1 example and 1 test prompt
        dataset a bigbench dataset that can be indexed by index. dataset[index]
        index: the index of the test example. This method will use the [index-nshots, index) datapoints as the nshot examples, wrapping if necessary
        returns: a string prompt, the answer as a string
        """
        #if index == -1:
        #    index = len(self.dataset['country'].tolist())-1
        country, gt = self.dataset['country'].tolist()[index], self.dataset['capital'].tolist()[index]
        inputs = f"""Q: What is the capital of {country}?\nA:"""
        #inputs = datapoint['inputs']#, datapoint['multiple_choice_targets']
        #gt = datapoint['targets'][0].title() #captialize the first letter!
        #gt_idx = targets.index(gt)
        prompt = inputs
        if include_answer:
            prompt+=' '+gt
        return prompt, ' '+gt


def generate_ans(model, prompter, idx):
    #making the prompt
    prompt, gt = prompter.nshot_open_prompt(nshots, idx)
    targets = [gt]#prompter.get_mc_targets(idx)
    gt_idx = targets.index(gt)
    #print(prompt, idx, gt)
    #running prompt thru model
    prompt_ids = model.tokenize(prompt)
    logits = model.get_layers(prompt_ids)

    probs_results = {}
    rrs_results   = {}

    for i in range(len(targets)):
        tgt = targets[i]
        probs, rrs = get_probs_and_mrrs(model, logits, tgt)
        probs_results[i] = probs.tolist()
        rrs_results[i] = rrs.tolist()

        if i == gt_idx:
            correct = int(rrs_results[i][-1]==1)

    top10_per_layer = model.topk_per_layer(logits, 10)
    prompt_results = {'inputs':prompt, 'targets':targets, 'answer':gt, 'answer_idx':gt_idx, 'probs':probs_results, 'rrs':rrs_results, 'top10_per_layer':top10_per_layer}
    #Lastly, turn the logits into a bunch of prob distributions over the whole vocab. This will be stored separately
    prob_distros = None#from_layer_logits_to_prob_distros(logits)
    return prompt_results, correct

def get_open_generations(model, dataset):
    global intervene, start_layer
    output = []
    prompter = CapitalsPrompter(dataset)
    console.print("LENGTH DATA:", len(prompter))
    total_correct = 0
    total = len(prompter)
    #all_prob_distros = []
    with torch.no_grad():
        for i in track(range(len(prompter)), description='iterating...'):
            json_out, correct = generate_ans(model, prompter, i)
            total_correct+=correct
            output.append(json_out)
            #all_prob_distros.append(prob_distros)

    if intervene == 'ovector':
        with open(f'{model_name}_{nshots}_open_caps_at{start_layer}interv_results.json', 'w') as fp:
            json.dump(output, fp, indent=4)
    elif intervene == 'ablate':
        with open(f'{model_name}_{nshots}_open_caps_at{start_layer}ablate_results.json', 'w') as fp:
            json.dump(output, fp, indent=4)
    else:
        with open(f'{model_name}_{nshots}_open_caps_results.json', 'w') as fp:
            json.dump(output, fp, indent=4)

    accuracy = float(total_correct)/float(total)
    return accuracy
    #all_prob_distros = np.stack(all_prob_distros)
    #print(all_prob_distros.shape)
    #np.save(f'{model_name}_{nshots}_open_caps_vocab_distros.npy', all_prob_distros)


if __name__ == "__main__":

    model_name = sys.argv[1]
    nshots = sys.argv[2]
    start_layer=19
    console.print(model_name, nshots, 'shot(s)')
    timer_task = timer.add_task("Loading model")
    intervene = None#'ovector'
    print("INTNERENE", intervene)
    with timer:
        if 'gpt2' in model_name:#model_name == 'gpt2-xl':
            model, tokenizer = load_gpt2(model_name)
            model = GPT2Wrapper(model, tokenizer)#GPTJWrapper(gptj, tokenizer)
            if intervene is not None:
                capital_ffn = torch.tensor(np.load('../city_capital_o19_gpt2-medium.npy')).cuda().half()
                for i in range(start_layer,24):
                    if intervene == 'ovector':
                        model.model.transformer.h[i].mlp = LambdaLayer(lambda x: capital_ffn)
                    elif intervene == 'ablate':
                        model.model.transformer.h[i].mlp = nn.Identity()
                console.print("Intervened at layer 19")
        elif 'gptj' == model_name:
            model, tokenizer = load_gptj()
            model = GPTJWrapper(model, tokenizer)
        elif 'bloom' in model_name:
            model, tokenizer = load_bloom(model_name)
            model = BloomWrapper(model, tokenizer)
    timer.stop_task(timer_task)
    timer.update(timer_task, visible=False)
 

    dataset = pd.read_csv("world_capitals.csv")
    accuracies = []
    #get_mc_generations(model, dataset)
    if ',' in nshots:
        rnshots = [int(n) for n in nshots.split(',')]
        for n in rnshots:
            console.print("NSHOTS", n)
            nshots = n
            acc = get_open_generations(model, dataset)
            console.print("Accuracy:", acc)
            accuracies.append(acc)
    else:
        nshots = int(nshots)
        get_open_generations(model, dataset)

    console.print("Final Accuracies:\n", accuracies)
