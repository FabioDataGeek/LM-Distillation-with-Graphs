import pickle as pkl
from utils import *

dataset = "ag-news"  # or "ag-news", etc.
merged_path = f"/usrvol/experiments/explainability_results/{dataset}_merged_skip_bigrams_sentences.pkl"

# Load the merged skipgram dictionary
with open(merged_path, "rb") as f:
    merged_skipgrams = pkl.load(f)

# Generate heatmaps with top_n=30, no numeric annotations (annot=False)
visualize_bigrams_heatmaps(merged_skipgrams, 
                            dataset=dataset,
                            top_n=20,
                            annot=False)