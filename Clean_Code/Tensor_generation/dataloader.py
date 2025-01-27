import torch
from torch.utils.data import Dataset
from torch_geometric.loader import DataLoader
from torch_geometric.utils.convert import from_networkx
from Embedding_generator import *

""" Si la oración es muy corta, puede ser que no se generen edge_labels para el grafo, con lo cual puede dar error 
a la hora de procesarlos, para evitar esto, vamos a añadir el elemento edge_label vacío en caso de que no se haya 
generado en el grafo."""

class CustomDataset(Dataset):
    def __init__(self, graph_list):
        new_graph_list = []
        for i in range(len(graph_list)):
            graph1 = graph_list[i][0]
            graph2 = graph_list[i][1]
            label = graph_list[i][2]
            data1 = from_networkx(graph1)
            data2 = from_networkx(graph2)
            if not hasattr(data1, 'edge_label'):
                data1.edge_label = torch.tensor([])
            if not hasattr(data2, 'edge_label'):
                data2.edge_label = torch.tensor([])
            
            new_graph_list.append((data1, data2, label))
        self.graph_list = new_graph_list


    def __len__(self):
        return len(self.graph_list)

    def __getitem__(self, idx):
        return self.graph_list[idx][0], self.graph_list[idx][1], self.graph_list[idx][2]