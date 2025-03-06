import pickle as pkl
import numpy as np
from tqdm import tqdm
from utils import *
from arguments import *
import os
import seaborn as sns
import matplotlib.pyplot as plt
import spacy
import re

nlp = spacy.load("en_core_web_sm")
stop_words = nlp.Defaults.stop_words
punctuations = nlp.Defaults.tokenizer_exceptions.keys()
symbols = {'!', '"', '#', '$', '%', '&', "'", '(', ')', '*', '+', ',', '-', '.', '/', ':', ';', '<', '=', '>', '?', '@', '[', '\\', ']', '^', '_', '`', '{', '|', '}', '~', '', ' '}

# Function to check if a word has semantic meaning
def is_valid_word(word):
    return len(word) > 1 and word.isalpha()

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

sentences = {}
labels = LABELS[dataset]
for label in labels:
    sentences[label] = {'correct': {}, 'incorrect': {}}

iterator = 0
folder = os.listdir(f"/usrvol/experiments/explainability_results/{dataset}")
for i in tqdm(range(len(folder)), desc="Processing files: "):
    with open(f"/usrvol/experiments/explainability_results/{dataset}/results_{i}.pkl", "rb") as f:
        explanations = pkl.load(f)
        
    for explanation_id, explanation in tqdm(explanations.items()):
        prediction = explanation[5]
        label = explanation[6]
        dict_nodes = explanation[3].dict_nodes[0]
        words = list(dict_nodes.values())
        filtered_words = [word for word in words if not '«' in word and not '»' in word]
        filtered_words = [word.lower() for word in filtered_words]
        filtered_words = [word for word in filtered_words if not word in stop_words]
        filtered_words = [word for word in filtered_words if not word in punctuations]
        filtered_words = [word for word in filtered_words if not word in symbols]
        filtered_words = [word for word in filtered_words if is_valid_word(word)]

        
        if label == prediction:
            sentences[label]['correct'][iterator] = filtered_words
        else:
            sentences[label]['incorrect'][iterator] = filtered_words
        
        iterator += 1
        
with open(f"/usrvol/experiments/explainability_results/{dataset}_sentences.pkl", "wb") as f:
    pkl.dump(sentences, f)


