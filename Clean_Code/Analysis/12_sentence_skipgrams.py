import pickle as pkl
import numpy as np
from tqdm import tqdm
from utils import *
from arguments import *

args = arguments()
dataset = "ag-news"

# Change this if you want skip size > 0 (0 = adjacent bigrams)
SKIP_SIZE = 5

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

with open(f"/usrvol/experiments/explainability_results/{dataset}_sentences.pkl", "rb") as f:
    sentences = pkl.load(f)
    
# skipgram_bag will hold frequency counts of skip-bigrams for each label + subset
skipgram_bag = {}
for label in LABELS[dataset]:
    skipgram_bag[label] = {
        'correct': {},
        'incorrect': {}
    }

for label in tqdm(LABELS[dataset], desc=f"Processing sentences from {dataset}: "):
    for subset in ["correct", "incorrect"]:
        sentences_dict = sentences[label][subset]
        # sentences_dict is expected to be something like:
        # { id1: [token1, token2, ...], id2: [...], ... }
        
        for sent_id, sentence_tokens in sentences_dict.items():
            # Generate skip-bigrams for this sentence
            # i.e., pairs of tokens separated by SKIP_SIZE tokens in between
            for i in range(len(sentence_tokens) - (SKIP_SIZE + 1)):
                first_token = sentence_tokens[i]
                second_token = sentence_tokens[i + SKIP_SIZE + 1]
                
                # Sort them to remove directional difference
                token_pair = sorted([first_token, second_token])  # e.g. ["dog", "the"]
                skipgram = "|".join(token_pair)                  # "dog|the"
                
                # Now store counts
                if skipgram not in skipgram_bag[label][subset]:
                    skipgram_bag[label][subset][skipgram] = 1
                else:
                    skipgram_bag[label][subset][skipgram] += 1

# Save the result
out_path = f"/usrvol/experiments/explainability_results/{dataset}_sentence_skipgrams_{SKIP_SIZE}_freq.pkl"
with open(out_path, "wb") as f:
    pkl.dump(skipgram_bag, f)

print(f"Skip-bigram frequencies saved to {out_path}.")
