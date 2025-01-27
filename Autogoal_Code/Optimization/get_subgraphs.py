import torch
from dig.xgraph.method import SubgraphX
import numpy as np
from tqdm import tqdm
from architecture_GNNs import *
from dataloader import *
import pickle as pkl
from datetime import datetime

args = arguments()
dataset_name = 'sst2'
def load_dataset():
    dataset = Dataset_GNN(
        root=args["root_test_data_path"], files_path=args["raw_test_data_path"]
    )
    return dataset

with open(f"/home/coder/autogoal/predictions_labels_reals_{dataset_name}.npy", "rb") as f:
    predictions_labels_reals = np.load(f)
    
# Load dataset
dataset = load_dataset()

num_hops = 2  # int
rollout = 300  # int
min_atoms = 1  # int
c_puct = 1  # int
expand_atoms = 5  # int
local_radius = 5  # int
sample_num = 5  # int
max_nodes = 2  # int

model = GNN_classifier(size=args['size'], num_layers=args['num_layers'], 
                       dropout=args['dropout'], module= args['module'], 
                       layer_norm=args['layer_norm'], residual=args['residual'], 
                       pooling=args['pooling'], lin_transform=args['lin_transform'])

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
        save_dir="/usrvol/explainability_results/",
    )

number_datasets = len(dataset)

iterator = 0

for i in tqdm(range(number_datasets), desc='Datasets', colour='green', leave=True):
    results_dict = {}
    subdataset = dataset[i]
    subdataset_length = len(subdataset)

    loader = HomogeneousDataLoader(subdataset, batch_size=1, shuffle=False)


    for j, batch in tqdm(enumerate(loader), total=len(loader), leave=True, colour='blue'):
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
        for k, result in enumerate(results):
            del result['data']
            del result['ori_graph']

            if result['coalition'] == related_pred['subgraph']:
                explanation_idx = k
    
        prediction = predictions_labels_reals[0][iterator]
        label = predictions_labels_reals[1][iterator]
        real = predictions_labels_reals[2][iterator]
        results_dict[j] = [results, related_pred, batch, subdataset[j], explanation_idx, prediction, label, real]
        iterator += 1
        
    if not os.path.exists(f'/home/coder/autogoal/explainability_results/'):
        os.makedirs(f'home/coder/autogoal/explainability_results/')

    with open(f'/home/coder/autogoal/explainability_results/results_{i}_{dataset_name}.pkl', 'wb') as f:
        pkl.dump(results_dict, f)