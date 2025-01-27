import pickle as pkl
import torch
from torch_geometric.data import Dataset, Data, DataLoader, Batch
from transformers import AutoTokenizer
import os
from torch_geometric.utils import add_remaining_self_loops, add_self_loops
from tqdm import tqdm
from dicts import *
import random
from torch.utils.data import Dataset as Dataset_torch
from torch.utils.data import DataLoader as DataLoader_torch
from torch_geometric.utils import to_undirected

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
from arguments import *

args = arguments()

""" from torch import Dataset as Dataset_torch """

""" The following Datasets developed for GNNs and LLMs can be used
to load a file at a time, it should be used with os.listdir() to 
load all the files in a directory. 
This apply for training, validation and test."""


LABEL_DICT = {'contradiction': 0, 'entailment': 1, 'neutral': 2, '-': 3}


# This dataset consider all relations the same, i.e. homogeneous graphs
class Dataset_GNN(Dataset):
    def __init__(self, root, files_path, transform=None, pre_transform=None):
        
        self.files_path = files_path
        super(Dataset_GNN, self).__init__(root, transform, pre_transform)

    @property
    def raw_file_names(self):
        return os.listdir(self.files_path)

    @property
    def processed_file_names(self):
        # The processed files are named after the raw files
        return self.raw_file_names

    def download(self):
        pass  # Implement this if your data is available online

    def process(self):
        for raw_path, processed_path in zip(self.raw_paths, self.processed_paths):
            with open(raw_path, 'rb') as f:
                graph = pkl.load(f)
            data_list = []
            for data, label in graph:
                label = [LABEL_DICT[l] for l in label]
                edge_index = data.edge_index
                dict_nodes = data.dict_nodes
                edge_attr = data.edge_attr
                data = Data(x=data.x.to(device), edge_index=edge_index.to(device), dict_nodes=dict_nodes, edge_attr=edge_attr.to(device), 
                            y=torch.tensor(label).to(device))
                data_list.append(data)

            for i, data in enumerate(data_list):
                if len(data.x) != data.edge_index.max().item()+1:
                    print(f"Error in the edge index {i} for file {processed_path}:" )
                    print("")
            torch.save(data_list, processed_path)

    def len(self):
        return len(self.processed_file_names)

    def get(self, idx):
        # Load only the file at the requested index
        data_list = torch.load(self.processed_paths[idx])
        return data_list

class Dataset_GNN_guided(Dataset):
    def __init__(self, root, files_path, transform=None, pre_transform=None):
        
        self.files_path = files_path
        super(Dataset_GNN_guided, self).__init__(root, transform, pre_transform)

    @property
    def raw_file_names(self):
        return os.listdir(self.files_path)

    @property
    def processed_file_names(self):
        # The processed files are named after the raw files
        return self.raw_file_names

    def download(self):
        pass  # Implement this if your data is available online

    def process(self):
        for raw_path, processed_path in zip(self.raw_paths, self.processed_paths):
            with open(raw_path, 'rb') as f:
                graph = pkl.load(f)
            data_list = []
            for data, _, lm_label in graph:
                edge_index = data.edge_index
                dict_nodes = data.dict_nodes
                edge_attr = data.edge_attr
                data = Data(x=data.x.to(device), edge_index=edge_index.to(device), dict_nodes=dict_nodes, edge_attr=edge_attr.to(device), 
                            y=torch.tensor(lm_label).to(device))
                data_list.append(data)

            for i, data in enumerate(data_list):
                if len(data.x) != data.edge_index.max().item()+1:
                    print(f"Error in the edge index {i} for file {processed_path}:" )
                    print("")
            torch.save(data_list, processed_path)

    def len(self):
        return len(self.processed_file_names)

    def get(self, idx):
        # Load only the file at the requested index
        data_list = torch.load(self.processed_paths[idx])
        return data_list

# This dataset consider all relations the same, i.e. homogeneous graphs
class Dataset_GNN_2graphs(Dataset):
    def __init__(self, root, files_path, transform=None, pre_transform=None):
        
        self.files_path = files_path
        super(Dataset_GNN_2graphs, self).__init__(root, transform, pre_transform)

    @property
    def raw_file_names(self):
        return os.listdir(self.files_path)

    @property
    def processed_file_names(self):
        # The processed files are named after the raw files
        return self.raw_file_names

    def download(self):
        pass  # Implement this if your data is available online

    def process(self):
        for raw_path, processed_path in zip(self.raw_paths, self.processed_paths):
            with open(raw_path, 'rb') as f:
                graph_pairs = pkl.load(f)
            data_list = []
            for data1, data2, label in graph_pairs:
                label = [LABEL_DICT[l] for l in label]
                edge_index1 = data1.edge_index
                edge_index2 = data2.edge_index
                dict_nodes1 = data1.dict_nodes
                dict_nodes2 = data2.dict_nodes
                edge_attr1 = data1.edge_attr
                edge_attr2 = data2.edge_attr
                data = Data(x1=data1.x.to(device), edge_index1=edge_index1.to(device), dict_nodes1=dict_nodes1, edge_attr1=edge_attr1.to(device), 
                            x2=data2.x.to(device), edge_index2=edge_index2.to(device), dict_nodes2=dict_nodes2, edge_attr2=edge_attr2.to(device),
                            y=torch.tensor(label)).to(device)
                data_list.append(data)

            for i, data in enumerate(data_list):
                if len(data.x1) != data.edge_index1.max().item()+1:
                    print(f"Error in the edge index {i} for file {processed_path}:" )
                    print("")
                if len(data.x2) != data.edge_index2.max().item()+1:
                    print(f"Error in the edge index {i} for file {processed_path}:" )
                    print("")
            torch.save(data_list, processed_path)

    def len(self):
        return len(self.processed_file_names)

    def get(self, idx):
        # Load only the file at the requested index
        data_list = torch.load(self.processed_paths[idx])
        return data_list


# This dataset considers the different types of relations i.e. heterogeneous graphs
class Dataset_RGNN(Dataset):
    def __init__(self, root, files_path, transform=None, pre_transform=None, sintactic=False, semantic=False, constituency=False):
        
        self.files_path = files_path
        self.sintactic = sintactic
        self.semantic = semantic
        self.constituency = constituency
        super(Dataset_RGNN, self).__init__(root, transform, pre_transform)

    @property
    def raw_file_names(self):
        return os.listdir(self.files_path)

    @property
    def processed_file_names(self):
        # The processed files are named after the raw files
        return self.raw_file_names

    def download(self):
        pass  # Implement this if your data is available online

    def process(self):
        for raw_path, processed_path in zip(self.raw_paths, self.processed_paths):
            with open(raw_path, 'rb') as f:
                graph = pkl.load(f)
            data_list = []
            for data, label in tqdm(graph):
                label = LABEL_DICT[label[0]]
                edge_index = data.edge_index
                try:
                    edge_label = data.edge_label[0]
                except:
                    edge_label = []
                # Only sintactic relations
                if self.sintactic and not self.semantic and not self.constituency:
                    for i in range(len(edge_label)):
                        edge_label[i] = SINTACTIC_NUM_DICT[SINTACTIC_DICT[edge_label[i]]]
                    self_loop = len(SINTACTIC_DICT)

                # Only semantic relations
                if self.semantic and not self.sintactic and not self.constituency:
                    for i in range(len(edge_label)):
                        edge_label[i] = SEMANTIC_NUM_DICT[SEMANTIC_DICT[edge_label[i]]]
                    self_loop = len(SEMANTIC_DICT)

                # Only constituency relations
                if self.constituency and not self.sintactic and not self.semantic:
                    for i in range(len(edge_label)):
                        edge_label[i] = CONSTITUENCY_NUM_DICT[CONSTITUENCY_DICT[edge_label[i]]]
                    self_loop = len(CONSTITUENCY_DICT)
                
                # Sintactic and semantic relations
                if self.sintactic and self.semantic and not self.constituency:
                    for i in range(len(edge_label)):
                        edge_label[i] = SIN_SEM_NUM_DICT[SIN_SEM_DICT[edge_label[i]]]
                    self_loop = len(SIN_SEM_DICT)

                # Sintactic and constituency relations
                if self.sintactic and self.constituency and not self.semantic:
                    for i in range(len(edge_label)):
                        edge_label[i] = SIN_CON_NUM_DICT[SIN_CON_DICT[edge_label[i]]]
                    self_loop = len(SIN_CON_DICT)

                # Semantic and constituency relations
                if self.semantic and self.constituency and not self.sintactic:
                    for i in range(len(edge_label)):
                        edge_label[i] = SEM_CON_NUM_DICT[SEM_CON_DICT[edge_label[i]]]
                    self_loop = len(SEM_CON_DICT)
                
                # Sintactic, semantic and constituency relations
                if self.sintactic and self.semantic and self.constituency:
                    for i in range(len(edge_label)):
                        edge_label[i] = SIN_SEM_CON_NUM_DICT[SIN_SEM_CON_DICT[edge_label[i]]]
                    self_loop = len(SIN_SEM_CON_DICT)
                
                
                edge_index_size = edge_index.size(1)
                edge_label_size = len(edge_label)
                if edge_label_size < edge_index_size:
                    edge_label1 += [self_loop]*(edge_index_size-edge_label_size)
                edge_label = torch.tensor(edge_label, dtype=torch.long)
                data = Data(x=data.x.to(device), edge_index=edge_index.to(device), edge_label=edge_label.to(device), y=torch.tensor(label).to(device))
                data_list.append(data)
            torch.save(data_list, processed_path)

    def len(self):
        return len(self.processed_file_names)

    def get(self, idx):
        # Load only the file at the requested index
        data_list = torch.load(self.processed_paths[idx], map_location=device)
        return data_list

# This dataset considers the different types of relations i.e. heterogeneous graphs
class Dataset_RGNN_guided(Dataset):
    def __init__(self, root, files_path, transform=None, pre_transform=None, sintactic=False, semantic=False, constituency=False):
        
        self.files_path = files_path
        self.sintactic = sintactic
        self.semantic = semantic
        self.constituency = constituency
        super(Dataset_RGNN_guided, self).__init__(root, transform, pre_transform)

    @property
    def raw_file_names(self):
        return os.listdir(self.files_path)

    @property
    def processed_file_names(self):
        # The processed files are named after the raw files
        return self.raw_file_names

    def download(self):
        pass  # Implement this if your data is available online

    def process(self):
        for raw_path, processed_path in zip(self.raw_paths, self.processed_paths):
            with open(raw_path, 'rb') as f:
                graph = pkl.load(f)
            data_list = []
            for data, _, lm_label in tqdm(graph):
                edge_index = data.edge_index
                try:
                    edge_label = data.edge_label[0]
                except:
                    edge_label = []
                # Only sintactic relations
                if self.sintactic and not self.semantic and not self.constituency:
                    for i in range(len(edge_label)):
                        edge_label[i] = SINTACTIC_NUM_DICT[SINTACTIC_DICT[edge_label[i]]]
                    self_loop = len(SINTACTIC_DICT)

                # Only semantic relations
                if self.semantic and not self.sintactic and not self.constituency:
                    for i in range(len(edge_label)):
                        edge_label[i] = SEMANTIC_NUM_DICT[SEMANTIC_DICT[edge_label[i]]]
                    self_loop = len(SEMANTIC_DICT)

                # Only constituency relations
                if self.constituency and not self.sintactic and not self.semantic:
                    for i in range(len(edge_label)):
                        edge_label[i] = CONSTITUENCY_NUM_DICT[CONSTITUENCY_DICT[edge_label[i]]]
                    self_loop = len(CONSTITUENCY_DICT)
                
                # Sintactic and semantic relations
                if self.sintactic and self.semantic and not self.constituency:
                    for i in range(len(edge_label)):
                        edge_label[i] = SIN_SEM_NUM_DICT[SIN_SEM_DICT[edge_label[i]]]
                    self_loop = len(SIN_SEM_DICT)

                # Sintactic and constituency relations
                if self.sintactic and self.constituency and not self.semantic:
                    for i in range(len(edge_label)):
                        edge_label[i] = SIN_CON_NUM_DICT[SIN_CON_DICT[edge_label[i]]]
                    self_loop = len(SIN_CON_DICT)

                # Semantic and constituency relations
                if self.semantic and self.constituency and not self.sintactic:
                    for i in range(len(edge_label)):
                        edge_label[i] = SEM_CON_NUM_DICT[SEM_CON_DICT[edge_label[i]]]
                    self_loop = len(SEM_CON_DICT)
                
                # Sintactic, semantic and constituency relations
                if self.sintactic and self.semantic and self.constituency:
                    for i in range(len(edge_label)):
                        edge_label[i] = SIN_SEM_CON_NUM_DICT[SIN_SEM_CON_DICT[edge_label[i]]]
                    self_loop = len(SIN_SEM_CON_DICT)
                
                
                edge_index_size = edge_index.size(1)
                edge_label_size = len(edge_label)
                if edge_label_size < edge_index_size:
                    edge_label1 += [self_loop]*(edge_index_size-edge_label_size)
                edge_label = torch.tensor(edge_label, dtype=torch.long)
                data = Data(x=data.x.to(device), edge_index=edge_index.to(device), edge_label=edge_label.to(device), y=torch.tensor(lm_label).to(device))
                data_list.append(data)
            torch.save(data_list, processed_path)

    def len(self):
        return len(self.processed_file_names)

    def get(self, idx):
        # Load only the file at the requested index
        data_list = torch.load(self.processed_paths[idx], map_location=device)
        return data_list


# This dataset considers the different types of relations i.e. heterogeneous graphs
class Dataset_RGNN_2graphs(Dataset):
    def __init__(self, root, files_path, transform=None, pre_transform=None, sintactic=False, semantic=False, constituency=False):
        
        self.files_path = files_path
        self.sintactic = sintactic
        self.semantic = semantic
        self.constituency = constituency
        super(Dataset_RGNN_2graphs, self).__init__(root, transform, pre_transform)

    @property
    def raw_file_names(self):
        return os.listdir(self.files_path)

    @property
    def processed_file_names(self):
        # The processed files are named after the raw files
        return self.raw_file_names

    def download(self):
        pass  # Implement this if your data is available online

    def process(self):
        for raw_path, processed_path in zip(self.raw_paths, self.processed_paths):
            with open(raw_path, 'rb') as f:
                graph_pairs = pkl.load(f)
            data_list = []
            for data1, data2, label in tqdm(graph_pairs):
                label = LABEL_DICT[label[0]]
                edge_index1 = data1.edge_index
                try:
                    edge_label1 = data1.edge_label[0]
                except:
                    edge_label1 = []
                edge_index2 = data2.edge_index
                try:
                    edge_label2 = data2.edge_label[0]
                except:
                    edge_label2 = []
                # Only sintactic relations
                if self.sintactic and not self.semantic and not self.constituency:
                    for i in range(len(edge_label1)):
                        edge_label1[i] = SINTACTIC_NUM_DICT[SINTACTIC_DICT[edge_label1[i]]]
                    for i in range(len(edge_label2)):
                        edge_label2[i] = SINTACTIC_NUM_DICT[SINTACTIC_DICT[edge_label2[i]]]
                    self_loop = len(SINTACTIC_DICT)

                # Only semantic relations
                if self.semantic and not self.sintactic and not self.constituency:
                    for i in range(len(edge_label1)):
                        edge_label1[i] = SEMANTIC_NUM_DICT[SEMANTIC_DICT[edge_label1[i]]]
                    for i in range(len(edge_label2)):
                        edge_label2[i] = SEMANTIC_NUM_DICT[SEMANTIC_DICT[edge_label2[i]]]
                    self_loop = len(SEMANTIC_DICT)

                # Only constituency relations
                if self.constituency and not self.sintactic and not self.semantic:
                    for i in range(len(edge_label1)):
                        edge_label1[i] = CONSTITUENCY_NUM_DICT[CONSTITUENCY_DICT[edge_label1[i]]]
                    for i in range(len(edge_label2)):
                        edge_label2[i] = CONSTITUENCY_NUM_DICT[CONSTITUENCY_DICT[edge_label2[i]]]
                    self_loop = len(CONSTITUENCY_DICT)
                
                # Sintactic and semantic relations
                if self.sintactic and self.semantic and not self.constituency:
                    for i in range(len(edge_label1)):
                        edge_label1[i] = SIN_SEM_NUM_DICT[SIN_SEM_DICT[edge_label1[i]]]
                    for i in range(len(edge_label2)):
                        edge_label2[i] = SIN_SEM_NUM_DICT[SIN_SEM_DICT[edge_label2[i]]]
                    self_loop = len(SIN_SEM_DICT)

                # Sintactic and constituency relations
                if self.sintactic and self.constituency and not self.semantic:
                    for i in range(len(edge_label1)):
                        edge_label1[i] = SIN_CON_NUM_DICT[SIN_CON_DICT[edge_label1[i]]]
                    for i in range(len(edge_label2)):
                        edge_label2[i] = SIN_CON_NUM_DICT[SIN_CON_DICT[edge_label2[i]]]
                    self_loop = len(SIN_CON_DICT)

                # Semantic and constituency relations
                if self.semantic and self.constituency and not self.sintactic:
                    for i in range(len(edge_label1)):
                        edge_label1[i] = SEM_CON_NUM_DICT[SEM_CON_DICT[edge_label1[i]]]
                    for i in range(len(edge_label2)):
                        edge_label2[i] = SEM_CON_NUM_DICT[SEM_CON_DICT[edge_label2[i]]]
                    self_loop = len(SEM_CON_DICT)
                
                # Sintactic, semantic and constituency relations
                if self.sintactic and self.semantic and self.constituency:
                    for i in range(len(edge_label1)):
                        edge_label1[i] = SIN_SEM_CON_NUM_DICT[SIN_SEM_CON_DICT[edge_label1[i]]]
                    for i in range(len(edge_label2)):
                        edge_label2[i] = SIN_SEM_CON_NUM_DICT[SIN_SEM_CON_DICT[edge_label2[i]]]
                    self_loop = len(SIN_SEM_CON_DICT)
                
                
                edge_index1_size = edge_index1.size(1)
                edge_index2_size = edge_index2.size(1)
                edge_label1_size = len(edge_label1)
                edge_label2_size = len(edge_label2)
                if edge_label1_size < edge_index1_size:
                    edge_label1 += [self_loop]*(edge_index1_size-edge_label1_size)
                if edge_label2_size < edge_index2_size:
                    edge_label2 += [self_loop]*(edge_index2_size-edge_label2_size)
                edge_label1 = torch.tensor(edge_label1, dtype=torch.long)
                edge_label2 = torch.tensor(edge_label2, dtype=torch.long)
                data = Data(x1=data1.x.to(device), edge_index1=edge_index1.to(device), edge_label1=edge_label1.to(device), 
                            x2=data2.x.to(device), edge_index2=edge_index2.to(device), edge_label2=edge_label2.to(device), 
                            y=torch.tensor(label)).to(device)
                data_list.append(data)
            torch.save(data_list, processed_path)

    def len(self):
        return len(self.processed_file_names)

    def get(self, idx):
        # Load only the file at the requested index
        data_list = torch.load(self.processed_paths[idx], map_location=device)
        return data_list

""" This dataset will be used for Gine, TransformerConv, NNConv and GENConv"""
class Dataset_edge_attr(Dataset):
    def __init__(self, root, files_path, transform=None, pre_transform=None, edge_tensors =''):
        
        self.files_path = files_path
        with open(edge_tensors, 'rb') as f:
            self.edge_tensors = pkl.load(f)
        super(Dataset_edge_attr, self).__init__(root, transform, pre_transform)

    @property
    def raw_file_names(self):
        return os.listdir(self.files_path)

    @property
    def processed_file_names(self):
        # The processed files are named after the raw files
        return self.raw_file_names

    def download(self):
        pass  # Implement this if your data is available online

    def process(self):
        for raw_path, processed_path in zip(self.raw_paths, self.processed_paths):
            with open(raw_path, 'rb') as f:
                graph_pairs = pkl.load(f)
            data_list = []
            for data1, data2, label in tqdm(graph_pairs):
                label = LABEL_DICT[label[0]]
                edge_index1 = data1.edge_index
                try:
                    edge_label1 = data1.edge_label[0]
                except:
                    edge_label1 = []
                edge_index2 = data2.edge_index
                try:
                    edge_label2 = data2.edge_label[0]
                except:
                    edge_label2 = []
                
                # For every type of relation, add the corresponding edge tensor
                edge_attr1 = []
                for i in range(len(edge_label1)):
                    edge_attr1.append(self.edge_tensors[edge_label1[i]])
                edge_attr2 = []
                for i in range(len(edge_label2)):
                    edge_attr2.append(self.edge_tensors[edge_label2[i]])
                edge_label1 = torch.stack(edge_label1)
                edge_label2 = torch.stack(edge_label2)
                data = Data(x1=data1.x.to(device), edge_index1=edge_index1.to(device), edge_label1=edge_label1.to(device), edge_attr1=edge_attr1.to(device), 
                            x2=data2.x.to(device), edge_index2=edge_index2.to(device), edge_label2=edge_label2.to(device), edge_attr2=edge_attr2.to(device), 
                            y=torch.tensor(label)).to(device)
                data_list.append(data)
            torch.save(data_list, processed_path)

    def len(self):
        return len(self.processed_file_names)

    def get(self, idx):
        # Load only the file at the requested index
        data_list = torch.load(self.processed_paths[idx], map_location=device)
        return data_list


""" def get_dicts(pytorch_data):
    x_dict = {}
    triples_dict = {}
    a = pytorch_data.edge_index[0][0:len(pytorch_data.edge_label[0])]
    b = pytorch_data.edge_index[1][0:len(pytorch_data.edge_label[0])]
    stacked = torch.stack([torch.tensor(a), torch.tensor(b)], dim=0)
    for i in range(len(stacked[0])):
                first, second = stacked[0][i].item(), stacked[1][i].item()
                first_string, second_string, relation = first.label[0][first], first.label[0][second], first.edge_label[0][i]
                if not first_string in x_dict:
                    position = pytorch_data.edge_index[0][i]
                    x_dict[first_string] = pytorch_data.x[position]
                
                if not second_string in x_dict:
                    position = pytorch_data.edge_index[1][i]
                    x_dict[second_string] = pytorch_data.x[position]
                if not (first_string, relation, second_string) in triples_dict:
                    triples_dict[first_string, relation, second_string] = torch.stack((stacked[0][i], stacked[1][i]), dim=0)
                triples_dict[first_string, relation, second_string] = 
    
    return x_dict, triples_dict """


""" Dataset used when the module needs edge tensors according to the edge_index (GINe and TransformerConv) """


""" This one is used for HANConv and HGTConv """
""" class Dataset_metadata(Dataset):
    def __init__(self, root, files_path, metadata, transform=None, pre_transform=None, edge_tensors ='', ):
        self.metadata = metadata
        self.files_path = files_path
        with open(edge_tensors, 'rb') as f:
            self.edge_tensors = pkl.load(f)
        super(Dataset_metadata, self).__init__(root, transform, pre_transform)

    @property
    def raw_file_names(self):
        return os.listdir(self.files_path)

    @property
    def processed_file_names(self):
        # The processed files are named after the raw files
        return self.raw_file_names

    def download(self):
        pass  # Implement this if your data is available online

    def process(self):
        for raw_path, processed_path in zip(self.raw_paths, self.processed_paths):
            with open(raw_path, 'rb') as f:
                graph_pairs = pkl.load(f)
            data_list = []
            for data1, data2, label in tqdm(graph_pairs):
                label = LABEL_DICT[label[0]]
                edge_index1 = data1.edge_index
                try:
                    edge_label1 = data1.edge_label[0]
                except:
                    edge_label1 = []
                edge_index2 = data2.edge_index
                try:
                    edge_label2 = data2.edge_label[0]
                except:
                    edge_label2 = []
                
                # For every type of relation, add the corresponding edge tensor
                for i in range(len(edge_label1)):
                    edge_label1[i] = self.edge_tensors[edge_label1[i]]
                for i in range(len(edge_label2)):
                    edge_label2[i] = self.edge_tensors[edge_label1[i]]

                edge_label1 = torch.stack(edge_label1)
                edge_label2 = torch.stack(edge_label2)
                data = Data(x1=data1.x.to(device), edge_index1=edge_index1.to(device), edge_label1=edge_label1.to(device), 
                            x2=data2.x.to(device), edge_index2=edge_index2.to(device), edge_label2=edge_label2.to(device), 
                            y=torch.tensor(label)).to(device)
                data_list.append(data)
            torch.save(data_list, processed_path)

    def len(self):
        return len(self.processed_file_names)

    def get(self, idx):
        # Load only the file at the requested index
        data_list = torch.load(self.processed_paths[idx], map_location=device)
        return data_list """

class Dataset_HeatConv(Dataset):
    def __init__(self, root, files_path, transform=None, pre_transform=None, sintactic=False, semantic=False, constituency=False, edge_attr=''):
        
        self.files_path = files_path
        self.sintactic = sintactic
        self.semantic = semantic
        self.constituency = constituency

        self.sintactics = list(SINTACTIC_DICT.keys())
        self.semantics = list(SEMANTIC_DICT.keys())
        with open(edge_attr, 'rb') as f:
            self.edge_attr = pkl.load(f)

        super(Dataset_HeatConv, self).__init__(root, transform, pre_transform)

    @property
    def raw_file_names(self):
        return os.listdir(self.files_path)

    @property
    def processed_file_names(self):
        # The processed files are named after the raw files
        return self.raw_file_names

    def download(self):
        pass  # Implement this if your data is available online

    def process(self):
        for raw_path, processed_path in zip(self.raw_paths, self.processed_paths):
            with open(raw_path, 'rb') as f:
                graph_pairs = pkl.load(f)
            data_list = []
            for data1, data2, label in tqdm(graph_pairs):
                label = LABEL_DICT[label[0]]
                edge_index1 = data1.edge_index
                try:
                    edge_label1 = data1.edge_label[0]
                except:
                    edge_label1 = []
                edge_index2 = data2.edge_index
                try:
                    edge_label2 = data2.edge_label[0]
                except:
                    edge_label2 = []
                
                
                # If we don't have constituency nodes we just have one type of nodes (words)
                if not self.constituency:
                    length1 = len(data1.label[0])
                    node_type1 = [0]*length1
                    length2 = len(data2.label[0])
                    node_type2 = [0]*length2

                    # If we have only sintactic or semantic relations, we don't need to differentiate the edge types
                    if (self.sintactic and not self.semantic) or (self.semantic and not self.sintactic):
                        length1 = len(data1.edge_label[0])
                        edge_type1 = [0]*length1
                        length2 = len(data2.edge_label[0])
                        edge_type2 = [0]*length2
                    
                    # if we have sintactic and semantic relations, we differentiate them using the dictionaries of relations
                    else:
                        edge_types1 = []
                        for i in range(len(data1.edge_label[0])):
                            if data1.edge_label[0][i] in self.sintactics:
                                edge_types1.append(0)
                            elif data1.edge_label[0][i] in self.semantics:
                                edge_types1.append(1)
                            else:
                                raise ValueError('Edge type not found')

                        edge_types2 = []
                        for i in range(len(data2.edge_label[0])):
                            if data2.edge_label[0][i] in self.sintactics:
                                edge_types2.append(0)
                            elif data2.edge_label[0][i] in self.semantics:
                                edge_types2.append(1)
                            else:
                                raise ValueError('Edge type not found')
                
                # We consider two different types of nodes (words and constituents)
                else:
                    node_type1 = []
                    for i in range(len(data1.label[0])):
                        if data1.label[0][i].startswith('«') and data1.label[0][i].endswith('»'):
                            node_type1.append(0)
                        else:
                            node_type1.append(1)  
                    node_type2 = []
                    for i in range(len(data2.label[0])):
                        if data2.label[0][i].startswith('«') and data2.label[0][i].endswith('»'):
                            node_type2.append(0)
                        else:
                            node_type2.append(1)

                    if not self.sintactic and not self.semantic:
                        length1 = len(data1.edge_label[0])
                        edge_type1 = [0]*length1
                        length2 = len(data2.edge_label[0])
                        edge_type2 = [0]*length2

                    elif self.sintactic and not self.semantic:
                        edge_type1 = []
                        for i in range(len(data1.edge_label[0])):
                            if data1.edge_label[0][i] == 'constituency relation':
                                edge_type1.append(1)
                            elif data1.edge_label[0][i] in self.sintactics:
                                edge_type1.append(0)
                            else:
                                raise ValueError('Edge type not found')

                        edge_type2 = []
                        for i in range(len(data2.edge_label[0])):
                            if data2.edge_label[0][i] == 'constituency relation':
                                edge_type2.append(1)
                            elif data2.edge_label[0][i] in self.sintactics:
                                edge_type2.append(0)
                            else:
                                raise ValueError('Edge type not found')
                    
                    elif self.semantic and not self.sintactic:
                        edge_type1 = []
                        for i in range(len(data1.edge_label[0])):
                            if data1.edge_label[0][i] == 'constituency relation':
                                edge_type1.append(1)
                            elif data1.edge_label[0][i] in self.semantics:
                                edge_type1.append(0)
                            else:
                                raise ValueError('Edge type not found')

                        edge_type2 = []
                        for i in range(len(data2.edge_label[0])):
                            if data2.edge_label[0][i] == 'constituency relation':
                                edge_type2.append(1)
                            elif data2.edge_label[0][i] in self.semantics:
                                edge_type2.append(0)
                            else:
                                raise ValueError('Edge type not found')
                        
                    elif self.sintactic and self.semantic:
                        edge_type1 = []
                        for i in range(len(data1.edge_label[0])):
                            if data1.edge_label[0][i] == 'constituency relation':
                                edge_type1.append(2)
                            elif data1.edge_label[0][i] in self.sintactics:
                                edge_type1.append(0)
                            elif data1.edge_label[0][i] in self.semantics:
                                edge_type1.append(1)
                            else:
                                raise ValueError('Edge type not found')

                        edge_type2 = []
                        for i in range(len(data2.edge_label[0])):
                            if data2.edge_label[0][i] == 'constituency relation':
                                edge_type2.append(2)
                            elif data2.edge_label[0][i] in self.sintactics:
                                edge_type2.append(0)
                            elif data2.edge_label[0][i] in self.semantics:
                                edge_type2.append(1)
                            else:
                                raise ValueError('Edge type not found')
                
                
                edge_type1 = torch.tensor(edge_type1, dtype=torch.long)
                edge_type2 = torch.tensor(edge_type2, dtype=torch.long)
                edge_attr1 = []
                edge_attr2 = []    
                for i in range(len(edge_label1)):
                    edge_attr1[i] = self.edge_attr[edge_label1[i]]
                for i in range(len(edge_label2)):
                    edge_attr2[i] = self.edge_attr[edge_label2[i]]
                
                edge_index1 = torch.stack([edge_index1[0][0:length1], 
                                        edge_index1[1][0:length1]], 
                                        dim=0)
                
                edge_index2 = torch.stack([edge_index2[0][0:length2], 
                                        edge_index2[1][0:length2]], 
                                        dim=0)
                    
                data = Data(x1=data1.x.to(device), edge_index1=edge_index1.to(device), edge_label1=edge_label1.to(device), edge_attr1=edge_attr1.to(device), edge_type1=edge_type1.to(device), node_type1=node_type1.to(device),
                            x2=data2.x.to(device), edge_index2=edge_index2.to(device), edge_label2=edge_label2.to(device), edge_attr2=edge_attr2.to(device), edge_type2=edge_type2.to(device), node_type2=node_type2.to(device),
                            y=torch.tensor(label)).to(device)
                data_list.append(data)
            torch.save(data_list, processed_path)

    def len(self):
        return len(self.processed_file_names)

    def get(self, idx):
        # Load only the file at the requested index
        data_list = torch.load(self.processed_paths[idx], map_location=device)
        return data_list


class Dataset_LLM(Dataset_torch):
    def __init__(self, file, tokenizer):
        with open(file, 'rb') as f:
            self.data = pkl.load(f)
            self.tokenizer = tokenizer
        

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sentence1 = self.data[idx][0]
        sentence2 = self.data[idx][1]
        inputs = self.tokenizer(sentence1, sentence2, return_tensors='pt', padding='max_length', truncation=True, max_length=300, return_attention_mask=True, return_token_type_ids=True)
        label = LABEL_DICT[self.data[idx][2]]
        
        return {
                'input_ids': inputs['input_ids'].squeeze(0),  # Remove the batch dimension
                'attention_mask': inputs['attention_mask'].squeeze(0),
                'token_type_ids': inputs['token_type_ids'].squeeze(0),
                'labels': torch.tensor(label)  # No need for an extra dimension
            }

### DATALOADERS


class HomogeneousDataLoader:
    def __init__(self, dataset, batch_size=1, shuffle=False):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle

    def __iter__(self):
        data = self.dataset
        if self.shuffle:
            random.shuffle(data)

        for i in range(0, len(data), self.batch_size):
            batch = data[i:i+self.batch_size]
            yield self.collate(batch)

    def collate(self, batch):
        # Implement your own collate function here
        x_list = [data.x for data in batch]
        if args['undirected']:
            edge_index_list = [to_undirected(data.edge_index) for data in batch]
        else:
            edge_index_list = [data.edge_index for data in batch]
        y_list = [data.y for data in batch]

        # Create a Batch object for each graph
        batch = Batch.from_data_list([Data(x=x1, edge_index=edge_index1) for x1, edge_index1 in zip(x_list, edge_index_list)])

        # Return a tuple of the two Batch objects and the target tensor
        return batch, torch.stack(y_list)

class HomogeneousDataLoader_2graphs:
    def __init__(self, dataset, batch_size=1, shuffle=False):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle

    def __iter__(self):
        data = self.dataset
        if self.shuffle:
            random.shuffle(data)

        for i in range(0, len(data), self.batch_size):
            batch = data[i:i+self.batch_size]
            yield self.collate(batch)

    def collate(self, batch):
        # Implement your own collate function here
        x1_list = [data.x1 for data in batch]
        if args['undirected']:
            edge_index1_list = [to_undirected(data.edge_index1) for data in batch]
        else:
            edge_index1_list = [data.edge_index1 for data in batch]
        x2_list = [data.x2 for data in batch]
        if args['undirected']:
            edge_index2_list = [to_undirected(data.edge_index2) for data in batch]
        else:
            edge_index2_list = [data.edge_index2 for data in batch]
        y_list = [data.y for data in batch]

        # Create a Batch object for each graph
        batch1 = Batch.from_data_list([Data(x=x1, edge_index=edge_index1) for x1, edge_index1 in zip(x1_list, edge_index1_list)])
        batch2 = Batch.from_data_list([Data(x=x2, edge_index=edge_index2) for x2, edge_index2 in zip(x2_list, edge_index2_list)])

        # Return a tuple of the two Batch objects and the target tensor
        return batch1, batch2, torch.stack(y_list)


class HeterogeneousDataLoader:
    def __init__(self, dataset, batch_size=1, shuffle=False):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle

    def __iter__(self):
        data = self.dataset
        if self.shuffle:
            random.shuffle(data)

        for i in range(0, len(data), self.batch_size):
            batch = data[i:i+self.batch_size]
            yield self.collate(batch)

    def collate(self, batch):
        # Implement your own collate function here
        x_list = [data.x for data in batch]
        edge_index_list = [data.edge_index for data in batch]
        edge_type_list = [data.edge_label for data in batch]

        y_list = [data.y for data in batch]

        # Create a Batch object for each graph
        batch = Batch.from_data_list([Data(x=x, edge_index=edge_index, edge_attr=edge_type) for x, edge_index, edge_type in zip(x_list, edge_index_list, edge_type_list)])

        # Return a tuple of the two Batch objects and the target tensor
        return batch, torch.stack(y_list)

class HeteroGeneousDataLoader_2graphs:
    def __init__(self, dataset, batch_size=1, shuffle=False):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle

    def __iter__(self):
        data = self.dataset
        if self.shuffle:
            random.shuffle(data)

        for i in range(0, len(data), self.batch_size):
            batch = data[i:i+self.batch_size]
            yield self.collate(batch)

    def collate(self, batch):
        # Implement your own collate function here
        x1_list = [data.x1 for data in batch]
        edge_index1_list = [data.edge_index1 for data in batch]
        edge_type1_list = [data.edge_label1 for data in batch]

        x2_list = [data.x2 for data in batch]
        edge_index2_list = [data.edge_index2 for data in batch]
        edge_type2_list = [data.edge_label2 for data in batch]
        
        
        y_list = [data.y for data in batch]

        # Create a Batch object for each graph
        batch1 = Batch.from_data_list([Data(x=x1, edge_index=edge_index1, edge_attr=edge_type1) for x1, edge_index1, edge_type1 in zip(x1_list, edge_index1_list, edge_type1_list)])
        batch2 = Batch.from_data_list([Data(x=x2, edge_index=edge_index2, edge_attr=edge_type2) for x2, edge_index2, edge_type2 in zip(x2_list, edge_index2_list, edge_type2_list)])

        # Return a tuple of the two Batch objects and the target tensor
        return batch1, batch2, torch.stack(y_list)
    
