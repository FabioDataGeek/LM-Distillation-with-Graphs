import pickle as pkl
import numpy as np
from tqdm import tqdm
from utils import *
from arguments import *
import spacy
import os

args = arguments()

dataset = "sst-2"

total_paths = {}
iterator = 0
folder = os.listdir(f"/usrvol/experiments/explainability_results/{dataset}")
for i in tqdm(range(len(folder)), desc="Processing files: "):
    with open(f"/usrvol/experiments/explainability_results/{dataset}/results_{i}.pkl", "rb") as f:
        explanations = pkl.load(f)
        
    for explanation_id, explanation in tqdm(explanations.items()):
        prediction = explanation[5]
        label = explanation[6]
        graph = to_networkx(explanation[3])
        dict_nodes = explanation[3].dict_nodes[0]
        
        
        new_graph = nx.DiGraph()
        new_graph.add_edges_from(graph.edges())
        new_graph.remove_edges_from(nx.selfloop_edges(graph))

        root_nodes = [node for node, in_degree in new_graph.in_degree() if in_degree == 0]
        leaf_nodes = [node for node, out_degree in new_graph.out_degree() if out_degree == 0]
        
        
        all_paths = []

        for root in root_nodes:
            for leaf in leaf_nodes:
                paths = list(nx.all_simple_paths(new_graph, source=root, target=leaf))
                all_paths.extend(paths)
        
        
        total_paths[iterator] = {'graph': new_graph, 'ids': dict_nodes, 'paths': all_paths, 'prediction': prediction, 'label': label}
        iterator += 1

with open(f"/usrvol/experiments/explainability_results/{dataset}_paths.pkl", "wb") as f:
    pkl.dump(total_paths, f)