import pickle as pkl
import numpy as np
from tqdm import tqdm
from utils import *
from arguments import *

args = arguments()
dataset = "ag-news"

LABELS = {
    "ag-news": [0, 1, 2, 3],
    "sst-2": [0, 1],
}


files = os.listdir(f"/usrvol/experiments/explainability_results/")
skip_files = [f for f in files if dataset in f]
skip_files = [f for f in skip_files if "freq" in f]


LABELS_MAPPER = {
    "ag-news": {
        0: "World",
        1: "Sports",
        2: "Business",
        3: "Sci-Tech"
    },
    "sst-2": {
        0: "Negative",
        1: "Positive"
    }
}

total_skipgrams = {}
labels = LABELS[dataset]
for label in labels:
    total_skipgrams[label] = {'correct': {}, 'incorrect': {}}

for file in skip_files:
    with open(f"/usrvol/experiments/explainability_results/{file}", "rb") as f:
        skipgrams_dict = pkl.load(f)
        for label in labels:
            for k, v in skipgrams_dict[label]['correct'].items():
                total_skipgrams[label]['correct'][k] = total_skipgrams[label]['correct'].get(k, 0) + v
            for k, v in skipgrams_dict[label]['incorrect'].items():
                total_skipgrams[label]['incorrect'][k] = total_skipgrams[label]['incorrect'].get(k, 0) + v
                    
    
with open(f"/usrvol/experiments/explainability_results/{dataset}_merged_skip_bigrams_sentences.pkl", "wb") as f:
    pkl.dump(total_skipgrams, f)