from pprint import pprint

import argparse
import json
import numpy as np
# Create an ArgumentParser object
parser = argparse.ArgumentParser()
#
## Define command-line arguments
parser.add_argument('filename', help='word to search')
args = parser.parse_args()
file = args.filename

with open(file, 'r') as f:
    data = json.load(f)
    pprint(data)
   

        



        