import pickle as pkl
import numpy as np
from tqdm import tqdm
from utils import *
from arguments import *
import spacy
import os

args = arguments()

dataset = "ag-news"

DATASET_LABELS = {
    "ag-news": [0, 1, 2, 3],
    "sst-2": [0, 1],
}
only_special_tokens = True

files = os.listdir(f"/usrvol/experiments/explainability_results/")
skip_files = [f for f in files if "skip_bigrams" in f]
skip_files = [f for f in skip_files if dataset in f]
if only_special_tokens:
    skip_files = [f for f in skip_files if "special_tokens" in f]
else:
    skip_files = [f for f in skip_files if not "special_tokens" in f]

merger = {}
for file in tqdm(skip_files, desc=f"Merging skip bigrams from {dataset}: "):
    with open(f"/usrvol/experiments/explainability_results/{file}", "rb") as f:
        skipgrams_dict = pkl.load(f)
        for label in DATASET_LABELS[dataset]:
            if label not in merger:
                merger[label] = {'correct': {}, 'incorrect': {}}
            for k, v in skipgrams_dict[label]['correct'].items():
                merger[label]['correct'][k] = merger[label]['correct'].get(k, 0) + v
            for k, v in skipgrams_dict[label]['incorrect'].items():
                merger[label]['incorrect'][k] = merger[label]['incorrect'].get(k, 0) + v
    

if only_special_tokens:
    with open(f"/usrvol/experiments/explainability_results/{dataset}_merged_skip_bigrams_special_tokens.pkl", "wb") as f:
        pkl.dump(merger, f)
else:
    with open(f"/usrvol/experiments/explainability_results/{dataset}_merged_skip_bigrams.pkl", "wb") as f:
        pkl.dump(merger, f)