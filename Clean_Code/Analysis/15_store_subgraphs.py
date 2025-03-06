import pickle as pkl
from utils import *
from tqdm import tqdm


dataset = "sst-2"

LABELS_MAPPER = {
    "ag-news": {0: "World", 1: "Sports", 2: "Business", 3: "Sci/Tech"},
    "sst-2": {0: "Negative", 1: "Positive"},
}

labels = LABELS_MAPPER[dataset]
results = {}

for label in labels.values():
    results[label] = {"correct": [], "incorrect": []}

folder = os.listdir(f"/usrvol/experiments/explainability_results/{dataset}/")
for file in tqdm(folder, desc="Processing files", leave=False, colour="blue"):
    if file.endswith(".pkl"):
        with open(f"/usrvol/experiments/explainability_results/{dataset}/{file}", "rb") as f:
            data = pkl.load(f)
        for idx, instance in tqdm(data.items(), desc="Processing Graphs", colour="green", leave=False):
            pyg_graph = instance[3]
            prediction = instance[5]
            label = instance[6]
            coalition = instance[1]['subgraph']
            subgraph = get_labeled_subgraph(pyg_graph, coalition)
            
            if prediction == label:
                results[labels[label]]["correct"].append(subgraph)
            else:
                results[labels[label]]["incorrect"].append(subgraph)
                
with open(f"/usrvol/experiments/explainability_results/{dataset}_nx_subgraphs.pkl", "wb") as f:
    pkl.dump(results, f)
            