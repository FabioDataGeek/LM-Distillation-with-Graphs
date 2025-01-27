import json
import os
from tqdm import tqdm
import pickle as pkl
import torch
from supar import Parser
import networkx as nx
import matplotlib.pyplot as plt
from nltk.tokenize import word_tokenize
from dataloader import *
from torch.utils.data import Dataset, DataLoader
from sintactic import *
from semantic import *
from constituency import *
import xml.etree.ElementTree as ET


def process_SNLI(folder:str,):
    files = os.listdir(folder)
    data_list = []
    filename_list = []
    for file in files:
        if file.endswith('.jsonl'):
            data = []
            filename = file.split('_')[2].split('.')[0]
            with open(folder + file, 'r') as f:
                for line in f:
                    data.append(json.loads(line))
            data_list.append(data)
            filename_list.append(filename)
    return data_list, filename_list


def process_RTE(folder, dataset):
    data_list = []
    filename_list = []
    files = os.listdir(folder)
    for file in files:
        if file.endswith("xml"):
            data = []
            filename = file.split('.')[0]
            with open(f"{folder}/{file}", "r") as f:
                tree = ET.parse(f)
            root = tree.getroot()
            data = []
            for child in root:
                for child2 in child:
                    for element in child2.iter('t'):
                        sentence1 = element.text
                    for element in child2.iter('h'):
                        sentence2 = element.text
                if dataset == 'RTE1':
                    info = {'sentence1': sentence1, 'sentence2': sentence2, 
                            'label': child.attrib['value'], 'id': child.attrib['id'],
                            'task': child.attrib['task']}
                if dataset == 'RTE2':
                    info = {'sentence1': sentence1, 'sentence2': sentence2, 
                            'label': child.attrib['entailment'], 'id': child.attrib['id'],
                            'task': child.attrib['task']}
                if dataset == 'RTE3':
                    info = {'sentence1': sentence1, 'sentence2': sentence2, 
                            'label': child.attrib['entailment'], 'id': child.attrib['id'],
                            'task': child.attrib['task'], 'length': child.attrib['length']}
                data.append(info)
            data_list.append(data)
            filename_list.append(filename)
    return data_list, filename_list


def process_SciTail(folder):
    files = os.listdir(folder)
    data_list = []
    filename_list = []
    for file in files:
        if file.endswith("txt"):
            data = []
            filename = file.split('_')[2].split('.')[0]
            with open(f"{folder}/{file}", 'r') as f:
                for line in f:
                    data.append(json.loads(line))
            data_list.append(data)
            filename_list.append(filename)
    return data_list, filename_list

def get_sentences(folder, dataset):
    """
    Get the sentences from the dataset.
    """

    if dataset == 'SNLI':
        data_list, filename_list = process_SNLI(folder)

    if dataset == 'RTE1' or dataset == 'RTE2' or dataset == 'RTE3':
        data_list, filename_list = process_RTE(folder, dataset)

    if dataset == "SciTail":
        data_list, filename_list = process_SciTail(folder)

    return data_list, filename_list


def build_graphs(dataset_name=None, sintactic=False, semantic=False, 
                 constituency=False, batch_size=8, sintactic_parser=None, 
                 semantic_parser=None, constituency_parser=None, 
                 data=None, filename=None):
    if dataset_name == 'SNLI':
        dataset = SNLIDataset(data)
    if dataset_name == 'RTE1' or dataset_name == 'RTE2' or dataset_name == 'RTE3':
        dataset = RTEDataset(data)
    if dataset_name == 'SciTail':
        dataset = SciTailDataset(data)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    iterator = 0
    for sentences1, sentences2, labels in tqdm(dataloader):
        processed_data = []

        if sintactic:
            syntactic_graphs1 = sintactic_parser.get_graph(sentences1)
            syntactic_graphs2 = sintactic_parser.get_graph(sentences2)
            for i in range(len(syntactic_graphs1)):
                processed_data.append((syntactic_graphs1[i], syntactic_graphs2[i], labels[i]))
            mode = 'sintactic'
        
        if semantic:
            semantic_graphs1 = semantic_parser.get_graph(sentences1)
            semantic_graphs2 = semantic_parser.get_graph(sentences2)
            for i in range(len(semantic_graphs1)):
                processed_data.append((semantic_graphs1[i], semantic_graphs2[i], labels[i]))
            mode = 'semantic'
        
        if constituency:
            constituency_graphs1 = constituency_parser.get_graph(sentences1)
            constituency_graphs2 = constituency_parser.get_graph(sentences2)
            for i in range(len(constituency_graphs1)):
                processed_data.append((constituency_graphs1[i], constituency_graphs2[i], labels[i]))
            mode = 'constituency'

        
        output_path = f"/usrvol/processed_data/{dataset_name}/{filename}" 
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        with open(f"{output_path}/{mode}{iterator}.pkl", 'wb') as f:
            pkl.dump(processed_data, f)
        iterator += 1


def get_graphs(data, filename, dataset, 
               sintactic=False, semantic=False, 
               constituency=False, knowledge=False, 
               batch_size=8):

    if sintactic:
        sintactic_parser = sintactic_graph_generator(model='dep-biaffine-roberta-en')
    else:
        sintactic_parser = None
    
    if semantic:
        semantic_parser = semantic_graph_generator(model='sdp-vi-en')
    else:
        semantic_parser = None

    if constituency:
        constituency_parser = constituency_graph_generator(model='con-crf-roberta-en')
    else:
        constituency_parser = None

    if knowledge:
        pass #Not Implemented

    build_graphs(dataset_name=dataset, sintactic=sintactic, semantic=semantic, constituency=constituency, 
                batch_size=batch_size, sintactic_parser=sintactic_parser, 
                semantic_parser=semantic_parser, constituency_parser=constituency_parser, 
                data=data, filename=filename)


if __name__ == '__main__':
    datasets = ['SNLI', 'RTE1', 'RTE2', 'RTE3','SciTail']
    for dataset in datasets:
        torch.cuda.empty_cache()
        folder = f"/usrvol/data/{dataset}/"            
        print(f"Processing {dataset} dataset...")
        data_list, filename_list = get_sentences(folder, dataset)

        for data, filename in zip(data_list, filename_list):
            print(f"Processing {filename}...")
            
            """ print("Generating sintactic graphs...")
            get_graphs(data, filename, dataset, sintactic=True, semantic=False, constituency=False, knowledge=False, batch_size=512)
            torch.cuda.empty_cache()

            print("Generating semantic graphs...")
            get_graphs(data, filename, dataset, sintactic=False, semantic=True, constituency=False, knowledge=False, batch_size=512)
            torch.cuda.empty_cache() """

            print("Generating constituency graphs...")
            get_graphs(data, filename, dataset, sintactic=False, semantic=False, constituency=True, knowledge=False, batch_size=512)    
            torch.cuda.empty_cache()