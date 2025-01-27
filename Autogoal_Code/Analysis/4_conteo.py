import pickle as pkl
import numpy as np
from tqdm import tqdm
import os
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from collections import Counter
import plotly.express as px
import pandas as pd
from utils import *
dataset = "sst2"

NUMBER_OF_LABELS = {
    "ag-news": 4,
    "sst2": 2,
}

# Main processing
for label in tqdm(range(NUMBER_OF_LABELS[dataset])):
    for prediction in ['correct', 'incorrect']:
        # Load triples
        with open(f"/home/coder/autogoal/labels/{dataset}/{label}/{prediction}/important_triples.pkl", "rb") as f:
            important_triples = pkl.load(f)
        with open(f"/home/coder/autogoal/labels/{dataset}/{label}/{prediction}/unimportant_triples.pkl", "rb") as f:
            unimportant_triples = pkl.load(f)

        # Count triples
        important_triple_count = Counter((triple[0], triple[2]) for triple in important_triples)
        unimportant_triple_count = Counter((triple[0], triple[2]) for triple in unimportant_triples)

        # Save triple plots
        save_triple_count_plot(
            important_triple_count, 10,
            f"/home/coder/autogoal/labels/{dataset}/{label}/{prediction}/important_triple_count.html",
            f"/home/coder/autogoal/labels/{dataset}/{label}/{prediction}/important_triple_count.pdf",
        )
        save_triple_count_plot(
            unimportant_triple_count, 10,
            f"/home/coder/autogoal/labels/{dataset}/{label}/{prediction}/unimportant_triple_count.html",
            f"/home/coder/autogoal/labels/{dataset}/{label}/{prediction}/unimportant_triple_count.pdf",
        )

        # Load words
        with open(f"/home/coder/autogoal/labels/{dataset}/{label}/{prediction}/important_words.pkl", "rb") as f:
            important_words = pkl.load(f)
        with open(f"/home/coder/autogoal/labels/{dataset}/{label}/{prediction}/unimportant_words.pkl", "rb") as f:
            unimportant_words = pkl.load(f)

        # Count words
        important_word_count = Counter(word.lower() for word in important_words)
        unimportant_word_count = Counter(word.lower() for word in unimportant_words)

        # Save word plots
        save_word_count_plot(
            important_word_count, 10,
            f"/home/coder/autogoal/labels/{dataset}/{label}/{prediction}/important_words.html",
            f"/home/coder/autogoal/labels/{dataset}/{label}/{prediction}/important_words.pdf",
            title="Top Important Words"
        )
        save_word_count_plot(
            unimportant_word_count, 10,
            f"/home/coder/autogoal/labels/{dataset}/{label}/{prediction}/unimportant_words.html",
            f"/home/coder/autogoal/labels/{dataset}/{label}/{prediction}/unimportant_words.pdf",
            title="Top Unimportant Words"
        )


