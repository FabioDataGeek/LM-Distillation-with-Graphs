import torch
from torch.utils.data import Dataset
from torch_geometric.loader import DataLoader
from torch_geometric.utils.convert import from_networkx
from Embedding_generator_constituency import *

""" Si la oración es muy corta, puede ser que no se generen edge_labels para el grafo, con lo cual puede dar error 
a la hora de procesarlos, para evitar esto, vamos a añadir el elemento edge_label vacío en caso de que no se haya 
generado en el grafo."""

class CustomDataset(Dataset):
    def __init__(self, graph_list):
        new_graph_list = []
        for i in range(len(graph_list[0][0])):
            graph = graph_list[0][0][i]
            label = graph_list[0][1][i]
            data = from_networkx(graph)
            if not hasattr(data, 'edge_label'):
                data.edge_label = torch.tensor([])
            new_graph_list.append((data, label))
        self.graph_list = new_graph_list


    def __len__(self):
        return len(self.graph_list)

    def __getitem__(self, idx):
        return self.graph_list[idx][0], self.graph_list[idx][1]