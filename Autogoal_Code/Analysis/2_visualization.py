import pickle as pkl
import numpy as np
from tqdm import tqdm
from utils import *
from arguments import *
import os

dataset = "ag-news"
with open(f"/home/coder/autogoal/explainability_results/summary_{dataset}.pkl", "rb") as f:
    Summary = pkl.load(f)

with open(f"/home/coder/autogoal/explainability_results/total_embeddings_{dataset}.pkl", "rb") as f:
    total_embeddings = pkl.load(f)

for key, value in tqdm(Summary.items(), desc="Processing instances"):
    number = key
    prediction = value['prediction'] # prediction
    label = value['label'] # label
    if not os.path.exists(f"/home/coder/autogoal/visualizations/{dataset}/{prediction}/correct"):
        os.makedirs(f"/home/coder/autogoal/visualizations/{dataset}/{prediction}/correct")
        
    if not os.path.exists(f"/home/coder/autogoal/visualizations/{dataset}/{prediction}/incorrect"):
        os.makedirs(f"/home/coder/autogoal/visualizations/{dataset}/{prediction}/incorrect")
        
    if prediction == label:
        visualize_word_embeddings_2d(Summary[number]['important_words'], Summary[number]['unimportant_words'], total_embeddings,
                                output_file=f"/home/coder/autogoal/visualizations/{dataset}/{prediction}/correct/word_embeddings_{dataset}_{number}.html",
                                pdf_file=f"/home/coder/autogoal/visualizations/{dataset}/{prediction}/correct/word_embeddings_{dataset}_{number}.pdf")
    else:
        visualize_word_embeddings_2d(Summary[number]['important_words'], Summary[number]['unimportant_words'], total_embeddings,
                                output_file=f"/home/coder/autogoal/visualizations/{dataset}/{prediction}/incorrect/word_embeddings_{dataset}_{number}.html",
                                pdf_file=f"/home/coder/autogoal/visualizations/{dataset}/{prediction}/incorrect/word_embeddings_{dataset}_{number}.pdf")