import torch
from dig.xgraph.method import SubgraphX
import numpy as np
from tqdm import tqdm
from experiment.Optimization.architecture_GNNs import *
from experiment.Clustered_optimization.dataloader import *
import pickle as pkl
from analysis_general import GraphAnalyzer_general
from datetime import datetime
from torch_geometric.data import Data
from experiment.Optimization.arguments import *

args = arguments()

def load_dataset():
    dataset = Dataset_GNN(
        root=args["root_test_data_path"], files_path=args["raw_test_data_path"]
    )
    return dataset

"""HIPERPARÁMETROS A OPTIMIZAR"""

num_hops = 2  # int
rollout = 300  # int
min_atoms = 1  # int
c_puct = 1  # int
expand_atoms = 5  # int
local_radius = 5 # int
sample_num = 5  # int
max_nodes = 2  # int
max_nodes_to_remove = 1  # int

dataset = load_dataset()
np.random.seed(54)
sample = np.random.choice(args['dataset_length'], args['num_samples'])
training_sample = []
for i in sample:
    specific_dataset = i // len(dataset[0])
    specific_graph = i % len(dataset[0])
    training_sample.append(dataset[specific_dataset][specific_graph])
model = GNN_classifier(
            size=args["size"],
            num_layers=args["num_layers"],
            dropout=args["dropout"],
            module=args["module"],
            layer_norm=args["layer_norm"],
            residual=args["residual"],
            pooling=args["pooling"],
            lin_transform=args["lin_transform"],
        )

explainer = SubgraphX(
    model=model,
    num_classes=args['labels'],
    device=device,
    num_hops=num_hops,
    rollout=rollout,
    min_atoms=min_atoms,
    c_puct=c_puct,
    expand_atoms=expand_atoms,
    local_radius=local_radius,
    sample_num=sample_num,
    max_nodes_to_remove=max_nodes_to_remove,
    save_dir="/usrvol/explainability_results/",
)
    
loader = HomogeneousDataLoader(training_sample, 1, shuffle=False)
results_dict = {}

for j, batch in tqdm(enumerate(loader), total=len(loader), leave=True):
    batch, label = batch
    data = Data(x=batch.x, edge_index=batch.edge_index, batch= batch.batch).to(device)
    
    # Graphs input information for subgraphX
    x = batch.x.to(device)
    edge_index = batch.edge_index.to(device)
    label = int(label)
    
    results, related_pred = explainer.explain(
        x=x,
        edge_index=edge_index,
        label=label,
        max_nodes=max_nodes,
    )
    
    explanation_idx = None
    for i, result in enumerate(results):
        del result['data']
        del result['ori_graph']

        if result['coalition'] == related_pred['subgraph']:
            explanation_idx = i           
    
    
    results_dict[sample[j]] = [results, related_pred, training_sample[j], explanation_idx]


with open(f"results_{datetime.now()}.pkl", "wb") as f:
    pkl.dump(results_dict, f)
