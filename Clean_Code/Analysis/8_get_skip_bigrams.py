import pickle as pkl
import numpy as np
from tqdm import tqdm
from utils import *
from arguments import *
import spacy
import os

args = arguments()

# Define your skip size here
SKIP_SIZE = 3  # 0 => bigrams, 1 => one token between them, etc.
only_special_tokens = True
dataset = "sst-2"

DATASET_LABELS = {
    "ag-news": [0, 1, 2, 3],
    "sst-2": [0, 1],
}

# Dictionary to store skipgrams counts
# We'll rename 'bigrams_dict' to 'skipgrams_dict' for clarity
skipgrams_dict = {}
for label in DATASET_LABELS[dataset]:
    skipgrams_dict[label] = {'correct': {}, 'incorrect': {}}

with open(f"/usrvol/experiments/explainability_results/{dataset}_paths.pkl", "rb") as f:
    total_paths = pkl.load(f)

for path_id in tqdm(total_paths, desc=f"Processing paths from {dataset}: "):
    paths = total_paths[path_id]['paths']
    ids = total_paths[path_id]['ids']
    prediction = total_paths[path_id]['prediction']
    label = total_paths[path_id]['label']

    if only_special_tokens:
        
        for path in paths:
            new_path = []
            for term in path:
                if '«' in ids[term] or '»' in ids[term]:
                    new_path.append(term)
                else:
                    continue
            paths[paths.index(path)] = new_path

    # For each path in this sample
    for path in paths:
        # Generate skipgrams
        skipgrams = []
        # We'll go up to len(path) - (SKIP_SIZE + 1) to avoid out-of-range
        for i in range(len(path) - (SKIP_SIZE + 1)):
            # The second index is i + SKIP_SIZE + 1
            first_token = ids[path[i]].lower()
            second_token = ids[path[i + SKIP_SIZE + 1]].lower()
            skipgram = "|".join([first_token, second_token])
            skipgrams.append(skipgram)

        # Classify skipgrams as correct/incorrect based on prediction
        if prediction == label:
            for sg in skipgrams:
                skipgrams_dict[label]['correct'][sg] = (
                    skipgrams_dict[label]['correct'].get(sg, 0) + 1
                )
        else:
            for sg in skipgrams:
                skipgrams_dict[label]['incorrect'][sg] = (
                    skipgrams_dict[label]['incorrect'].get(sg, 0) + 1
                )

# Save the skipgrams
if only_special_tokens:
    with open(f"/usrvol/experiments/explainability_results/{dataset}_skip_bigrams_{SKIP_SIZE}_special_tokens.pkl", "wb") as f:
        pkl.dump(skipgrams_dict, f)
else:
    with open(f"/usrvol/experiments/explainability_results/{dataset}_skip_bigrams_{SKIP_SIZE}.pkl", "wb") as f:
        pkl.dump(skipgrams_dict, f)

print("Skipgrams generated and saved successfully.")