import pickle as pkl
import numpy as np
from tqdm import tqdm
from utils import *
from arguments import *
import os
import seaborn as sns
import matplotlib.pyplot as plt


args = arguments()
dataset = "sst-2"
label = 0
LABELS = {
    "ag-news": [0, 1, 2, 3],
    "sst-2": [0, 1],
}

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

subsets = ["correct", "incorrect"]

only_special_tokens = False



if only_special_tokens:
    with open(f"/usrvol/experiments/explainability_results/{dataset}_merged_skip_bigrams_special_tokens.pkl", "rb") as f:
        merged_skipgrams = pkl.load(f)
else:
    with open(f"/usrvol/experiments/explainability_results/{dataset}_merged_skip_bigrams.pkl", "rb") as f:
        merged_skipgrams = pkl.load(f)
    
    
labels = LABELS[dataset]
for label in labels:
    for subset in subsets:
        
        
        skipgram_dict = merged_skipgrams[label][subset]
        
        # Build the matrix
        matrix, token2idx, idx2token = build_cooccurrence_matrix(skipgram_dict)
        
        matrix = matrix / matrix.sum()
        if only_special_tokens:
            title = f"Co-occurrence Matrix for {dataset}, label {LABELS_MAPPER[dataset][label]}, {subset} classifications, only special tokens"
        else:
            title = f"Co-occurrence Matrix for {dataset}, label {LABELS_MAPPER[dataset][label]}, {subset} classifications"
        visualize_cooccurrence(matrix, idx2token, title=title, top_n=5)