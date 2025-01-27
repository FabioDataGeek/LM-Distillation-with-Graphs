import torch
import os
import pickle as pkl
import networkx as nx


def get_triples(pytorch_data, entities, triples):
    a = pytorch_data.edge_index[0][0:len(pytorch_data.edge_label[0])]
    b = pytorch_data.edge_index[1][0:len(pytorch_data.edge_label[0])]
    stacked = torch.stack([torch.tensor(a), torch.tensor(b)], dim=0)
    for i in range(len(stacked[0])):
                first, second = stacked[0][i].item(), stacked[1][i].item()
                first_string, second_string, relation = first.label[0][first], first.label[0][second], first.edge_label[0][i]
                if not first_string in entities:
                    entities.append(first_string)
                if not second_string in entities:
                    entities.append(second_string)
                triples.append((first_string, relation, second_string))

    return list(set(entities)), list(set(triples))


entities = []
triples = []
for dataset in ['train', 'dev,', 'test']:
    folder = os.listdir(f"/usrvol/processed_tensors/SNLI/{dataset}/sintactic/bert-base-uncased/raw/")
    for file in folder:
        with open(f"/usrvol/processed_tensors/SNLI/{dataset}/sintactic/bert-base-uncased/raw/{file}", 'rb') as f:
            tensors = pkl.load(f)
        for tensor in tensors:
            first = tensor[0]
            entities, triples = get_triples(first, entities, triples)
            second = tensor[1]
            entities, triples = get_triples(second, entities, triples)

metadata = [entities, triples]