import json
import re
from tqdm import tqdm
import zstandard as zstd
from functools import partial
from pprint import pprint

import argparse

# Create an ArgumentParser object
parser = argparse.ArgumentParser()
#
## Define command-line arguments
parser.add_argument('--start', help='word to search')
#parser.add_argument('word', help='word to search')
#
args = parser.parse_args()

start = args.start
    
def findword(word, paragraph): # The word you want to find (case-insensitive)
    occurrences = re.findall(r'\b' + re.escape(word) + r'\b', paragraph, re.IGNORECASE)
    return len(occurrences)


def generate_examples(file, country, word):
   
    with open(f'subset_0/00.json', 'r') as f:
        filtered = json.load(f) ## this is a list of key

    key = 1
    occurence = 0
    occurence_all = 0
    occ_three = 0
    with zstd.open(open(file, "rb"), "rt", encoding="utf-8") as f:
        for row in f:

            if key < 3500000+1:              
                data = json.loads(row)
                se = [match.group(0).lower() for match in re.finditer(r'\b'+'beijing'+r'\b|\b'+'gaegae'+r'\b', data['text'], re.IGNORECASE)]
                m = re.finditer(r'\b'+'beijing'+r'\b', data['text'], re.IGNORECASE)
                se_all = [match.group(0) for match in m]
                if len(se_all) > 0:
                    print(se_all)
                se_all_three = findword(word, data['text'])
                #se = [a for a in se1 if a == ' beijing ']
                #if len(se1) is not 0:
                #    print("1 ", key, se1, flush = True)
                
                #se2 = [match.group(0).lower() for match in re.finditer(' beijing ', data['text'], re.IGNORECASE)]
                #se2 = [a for a in se2 if a == ' beijing ']
                #if len(se2) is not 0:
                #    print("2 ",key, se2, flush = True)
                #
                #if len(se1) != len(se2)

                occurence += len([a for a in se if 'beijing' == a])
                occurence_all += len(se_all)
                occ_three += se_all_three

                if len(se_all)!=  len([i for i in se_all if i == ' beijing ']):
                    print(se_all)
                if key % 100000 == 0:
                    print(file, key, occurence_all)
                    with open(f'test2.json', 'r') as f2:
                        d = json.load(f2)
                        d[key] = occurence
                    with open(f'test2.json', 'w') as f2:
                        json.dump(d, f2)
                    
                    with open(f'test.json', 'r') as f3:
                        d = json.load(f3)
                        d[key] = occurence_all
                    with open(f'test.json', 'w') as f3:
                        json.dump(d, f3)
                    
                    with open(f'test3.json', 'r') as f3:
                        d = json.load(f3)
                        d[key] = occ_three
                    with open(f'test3.json', 'w') as f3:
                        json.dump(d, f3)
            key += 1
    print(occurence)


generate_examples('/gpfs/data/epavlick/datasets/pile/train/00.jsonl.zst', 'a', 'beijing')

#generate_examples(["/gpfs/data/epavlick/datasets/pile/train/29.jsonl.zst"], "this")

#generate_examples(["data.1.jsonl.zst", "data.1.jsonl.zst", "data.1.jsonl.zst"], "that")

def make_files():
    output_file = 'data.1.jsonl'
    output_handle = open(output_file, 'w')
    import zstandard as zstd

    # Write JSONL records to the file
    output_handle.write(json.dumps({'text': 'This that how good'}) + '\n')
    output_handle.write(json.dumps({'text': 'that it is a good day'}) + '\n')
    output_handle.write(json.dumps({'text': 'that it is a good day'}) + '\n')
    output_handle.write(json.dumps({'text': 'that it is a good day'}) + '\n')
    # Add more output_handle.write() calls as needed for additional records

    output_handle.close()

    compressed_output_file = 'data.1.jsonl.zst'
    with open(output_file, 'rb') as input_handle:
        compressed_data = zstd.compress(input_handle.read(), level=3)

    with open(compressed_output_file, 'wb') as compressed_handle:
        compressed_handle.write(compressed_data)

def process(file_paths, country, word):
    
    from multiprocessing import Pool
    def process_files(file_paths, country, word):
        # Create a multiprocessing pool with the number of desired processes
        pool = Pool(processes=len(file_paths))

        # Map the process_file function to each file path in parallel
        with Pool() as pool:
            # Use starmap to pass multiple arguments to the process_file function
            pool.starmap(generate_examples, [(file_path, country, word) for file_path in file_paths])
    
    process_files(file_paths, country, word)


    # List of file paths to process
file_paths = [
'/gpfs/data/epavlick/datasets/pile/train/00.jsonl.zst',
 '/gpfs/data/epavlick/datasets/pile/train/01.jsonl.zst',
 '/gpfs/data/epavlick/datasets/pile/train/02.jsonl.zst',
 '/gpfs/data/epavlick/datasets/pile/train/03.jsonl.zst',
 '/gpfs/data/epavlick/datasets/pile/train/04.jsonl.zst',
 '/gpfs/data/epavlick/datasets/pile/train/05.jsonl.zst',
 '/gpfs/data/epavlick/datasets/pile/train/06.jsonl.zst',
 '/gpfs/data/epavlick/datasets/pile/train/07.jsonl.zst',
 '/gpfs/data/epavlick/datasets/pile/train/08.jsonl.zst',
 '/gpfs/data/epavlick/datasets/pile/train/09.jsonl.zst',
 '/gpfs/data/epavlick/datasets/pile/train/10.jsonl.zst',
 '/gpfs/data/epavlick/datasets/pile/train/11.jsonl.zst',
 '/gpfs/data/epavlick/datasets/pile/train/12.jsonl.zst',
 '/gpfs/data/epavlick/datasets/pile/train/13.jsonl.zst',
 '/gpfs/data/epavlick/datasets/pile/train/14.jsonl.zst',
 '/gpfs/data/epavlick/datasets/pile/train/15.jsonl.zst',
 '/gpfs/data/epavlick/datasets/pile/train/16.jsonl.zst',
 '/gpfs/data/epavlick/datasets/pile/train/17.jsonl.zst',
 '/gpfs/data/epavlick/datasets/pile/train/18.jsonl.zst',
 '/gpfs/data/epavlick/datasets/pile/train/19.jsonl.zst',
 '/gpfs/data/epavlick/datasets/pile/train/20.jsonl.zst',
 '/gpfs/data/epavlick/datasets/pile/train/21.jsonl.zst',
 '/gpfs/data/epavlick/datasets/pile/train/22.jsonl.zst',
 '/gpfs/data/epavlick/datasets/pile/train/23.jsonl.zst',
 '/gpfs/data/epavlick/datasets/pile/train/24.jsonl.zst',
 '/gpfs/data/epavlick/datasets/pile/train/25.jsonl.zst',
 '/gpfs/data/epavlick/datasets/pile/train/26.jsonl.zst',
 '/gpfs/data/epavlick/datasets/pile/train/27.jsonl.zst',
 '/gpfs/data/epavlick/datasets/pile/train/28.jsonl.zst',
 '/gpfs/data/epavlick/datasets/pile/train/29.jsonl.zst']

#################### Generate subset ##################
import csv

#process(file_paths, (args.country), (args.word).lower())
def exists_words(data):
    file_path = 'world_capitals.csv'  # Path to the CSV file
    word = []
    with open(file_path, 'r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            if row['country'].lower() == "cocos (keeling) islands":
                word.append(" cocos islands ")
            if row['country'].lower() == "republic of china (taiwan)":
                word.append(" taiwan ")
                word.append(" republic of china ")
            if row['country'].lower() == 'east timor (timor-leste)':
                word.append(" east timor ")
                word.append(" timor-leste ")
            if row['country'].lower() == 'united kingdom; england':
                word.append(" united kingdom ")
                word.append(" england ")
            word.append(' '+row['capital'].lower()+' ')
            word.append(' '+row['country'].lower()+' ')

    pattern = '|'.join(word)  # Combines the words with the OR operator
    return re.search(pattern, data, re.IGNORECASE)

def generate_subset_examples(file):
    key = 0
    new_name = file[file.rfind("/")+1: file.index('.')]
    con = []
    with zstd.open(open(file, "rb"), "rt", encoding="utf-8") as f:
        for row in f:
            key += 1
            if key > int(start) * 3500000 and key < (int(start) + 1) * 3500000 + 1:
            #if key > 4340000:
                data = json.loads(row)
                if exists_words(data['text']):
                    con.append(key)
                if key % 10000 == 0:
                    print(con)
                    try:
                        with open(f'subset_{start}/{new_name}.json', 'r') as f2:
                            d = json.load(f2)
                            d.extend(con)
                            con = []
                        with open(f'subset_{start}/{new_name}.json', 'w') as f2:
                            json.dump(d, f2)
                    except FileNotFoundError:
                        with open(f'subset_{start}/{new_name}.json', 'w') as f2:
                            json.dump(con, f2)
                            con = []
                    



def process_subset(file_paths):
    
    from multiprocessing import Pool
    
    def process_files(file_paths):
        # Create a multiprocessing pool with the number of desired processes
        pool = Pool(processes=len(file_paths))

        # Map the process_file function to each file path in parallel
        with Pool() as pool:
            # Use starmap to pass multiple arguments to the process_file function
            pool.map(generate_subset_examples, file_paths)
    
    process_files(file_paths)

#generate_subset_examples('/gpfs/data/epavlick/datasets/pile/train/00.jsonl.zst')
#process_subset(file_paths)

#generate_subset_examples('/gpfs/data/epavlick/datasets/pile/train/00.jsonl.zst',)
#################### Search Subset for Cross Occurences ##################
from transformers import AutoTokenizer
def search_cooccurence(seq, tokenizer = AutoTokenizer.from_pretrained(f'EleutherAI/pythia-1b')):
    
    with open(f'subset/{seq}.json', 'r') as file:
        filtered = json.load(file) ## this is a list of key
    
    key = 0
    with zstd.open(open(f'/gpfs/data/epavlick/datasets/pile/train/{seq}.jsonl.zst', "rb"), "rt", encoding="utf-8") as f:
        for row in f:
            key += 1
            if key in filtered:
                data = json.loads(row)
                print(exists_words(data))

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
#for i in seq:
#    with open(f'subset/{i}.json', 'r') as file:
#            filtered = json.load(file) ## this is a list of key          
#            print(np.max(filtered))
#            print(len(filtered))
#            print()

#generate_subset_examples('/gpfs/data/epavlick/datasets/pile/train/00.jsonl.zst')