import pickle as pkl
import numpy as np
from tqdm import tqdm
from utils import *
import os

dataset = "sst-2"

DATASET_LABELS = {"ag-news": {0: "World", 1: "Sports", 2: "Business", 3: "Sci/Tech", 
                              4: "Wrong World", 5: "Wrong Sports", 6: "Wrong Business", 7: "Wrong Sci/Tech"},
                  "sst2": {0: "Negative", 1: "Positive", 2: "Wrong Negative", 3: "Wrong Positive"}}


NUMBER_OF_LABELS = {
    
    "ag-news": 4,
    "sst-2": 2
}



folder = os.listdir(f"/usrvol/experiments/explainability_results/{dataset}/")

label_list = []
properties_dict = {}
iterator = 0
for i in tqdm(range(len(folder)), desc="Extracting properties for each subdataset", colour="green"):
    with open(f"/usrvol/experiments/explainability_results/{dataset}/results_{i}.pkl", "rb") as f:
        explanations = pkl.load(f)
    number_of_labels = NUMBER_OF_LABELS[dataset]
    for explanation_id, explanation in tqdm(explanations.items(), desc="Extracting properties for each graph"):    
        non_difference = False
        graph = to_networkx(explanation[3], remove_self_loops=True, to_undirected=False)
        graph = eliminate_cross_links_and_isolated_nodes(graph)
        coalition = explanation[1]['subgraph']
        subgraph, min_distance = get_subgraph(graph, coalition)
        subgraph_root = sorted(subgraph.nodes)[0]
        subgraph = eliminate_cross_links_and_isolated_nodes(subgraph, root=subgraph_root)
        difference_graph = graph_difference(graph, subgraph)
        try:
            difference_root = sorted(difference_graph.nodes)[0]
            difference_graph = eliminate_cross_links_and_isolated_nodes(difference_graph, root=0)
        except:
            non_difference = True
        
        # if explanation_id >= len(predictions):
        #     continue
        
        prediction = explanation[5]
        label = explanation[6]
        
        for modality in ['graph', 'subgraph', 'difference_graph']:
            
            if modality == 'graph':
                objective = graph
            elif modality == 'subgraph':
                objective = subgraph
            else:
                if non_difference:
                    continue
                objective = difference_graph
        
            if not os.path.exists(f"/usrvol/experiments/properties/{dataset}/{prediction}/{modality}/correct"):
                os.makedirs(f"/usrvol/experiments/properties/{dataset}/{prediction}/{modality}/correct/", exist_ok=True)
            
            if not os.path.exists(f"/usrvol/experiments/properties/{dataset}/{prediction}/incorrect"):
                os.makedirs(f"/usrvol/experiments/properties/{dataset}/{prediction}/{modality}/incorrect", exist_ok=True)
            
            if not label in label_list:
                label_list.append(label)
            
            if label == prediction:
                properties_dict[prediction] = {'correct': {'graph':{}, 'subgraph': {}, 'difference_graph': {}}, 'incorrect': {'graph':{}, 'subgraph': {}, 'difference_graph': {}}} if properties_dict.get(prediction) is None else properties_dict[prediction]
                properties_dict[prediction]['correct'][modality][iterator] = get_graph_properties(objective, max_size=5)
                
            else:
                properties_dict[prediction] = {'correct': {'graph':{}, 'subgraph': {}, 'difference_graph': {}}, 'incorrect': {'graph':{}, 'subgraph': {}, 'difference_graph': {}}} if properties_dict.get(prediction) is None else properties_dict[prediction]
                properties_dict[prediction]['incorrect'][modality][iterator] = get_graph_properties(objective, max_size=5)

        iterator += 1

for label in label_list:
    for modality in ['graph', 'subgraph', 'difference_graph']:
        with open(f"/usrvol/experiments/properties/{dataset}/{label}/{modality}/correct/properties.pkl", "wb") as f:
            pkl.dump(properties_dict[label]['correct'][modality], f)
            
        with open(f"/usrvol/experiments/properties/{dataset}/{label}/{modality}/incorrect/properties.pkl", "wb") as f:
            pkl.dump(properties_dict[label]['incorrect'][modality], f)