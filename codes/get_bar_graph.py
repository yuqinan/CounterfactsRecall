import json

def generate_include():
    with open(f'/gpfs/data/epavlick/overwrite/pythia-2.8b_decoded_without_counters.json', 'r') as f:
        original = json.load(f)
        clean = []
        for idx, i in enumerate(original):
            if i['gt'] in i['decoded']:
                clean.append(i['input_para'])
    return clean

clean = generate_include()


def one_scale():
    with open(f'final_pythia-1.4b_counterfacts_camera_ready.json', 'r') as f:
        all_data = json.load(f)
    print(len(all_data))

    #clean_input = []
    #for i in all_data:
    #    try:
    #        if i['input_para'][i['input_para'].index('.')+2:] in clean:
    #            clean_input.append(i)
    #        
    #    except ValueError:
    #        continue
    #        print(i['input'])
           
    #all_data = clean_input
    print(len(all_data))

    scale = [-1.5, 10.0, 1.5, -6.0]

    all_result = {}
    all_result["1"] = {'ct': 0, 'gt': 0}
    for s in scale:
        all_result[str(s)] = {'ct': 0, 'gt': 0}

    all_result['1']['ct'] = len([i for i in all_data if i['correct'] == 1]) / len(all_data)
    all_result['1']['gt'] = len([i for i in all_data if i['gt_correct'] == 1]) / len(all_data)

    for s in scale:
        all_result[str(s)]['ct'] = len([i for i in all_data if i[f'intervened_correct_{s}'] == 1]) / len(all_data)
        all_result[str(s)]['gt'] = len([i for i in all_data if i[f'intervened_gt_correct_{s}'] == 1])/ len(all_data)
    print(len(all_result))
    print(all_result)

def both_scale():
    with open('final_pythia-2.8b.json', 'r') as f:
        all_data = json.load(f)
    print(len(all_data))

    scale = [[-2.8, 10.0], [2.6, -1.5]]
    #scale = [[-0.7, 10.0], [3.5, -6.0]]
    all_result = {}
    all_result["1"] = {'ct': 0, 'gt': 0}
    for s in scale:
        all_result[f'{str(s[0])}_{str(s[1])}'] = {'ct': 0, 'gt': 0}
    all_data = [i for i in all_data if i['correct'] !=3]
    all_result['1']['ct'] = len([i for i in all_data if i['correct'] == 1]) / len(all_data)
    all_result['1']['gt'] = len([i for i in all_data if i['gt_correct'] == 1]) / len(all_data)

    for s in scale:
        all_result[f'{str(s[0])}_{str(s[1])}']['ct'] = len([i for i in all_data if i[f'intervened_correct_{s[0]}_{s[1]}'] == 1]) / len(all_data)
        all_result[f'{str(s[0])}_{str(s[1])}']['gt'] = len([i for i in all_data if i[f'intervened_gt_correct_{s[0]}_{s[1]}'] == 1])/ len(all_data)

    print(all_result)

one_scale()