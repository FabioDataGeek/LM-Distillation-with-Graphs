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
from datasets import load_dataset


ds = load_dataset("stanfordnlp/sst2")



def build_graphs(dataset, subset, parser, batch_size=8):

    instance = load_dataset(dataset, split=subset)
    instance.set_format(type='torch')
    dataloader = DataLoader(dataset=instance, batch_size=batch_size, shuffle=False)
    iterator = 0
    for batch in tqdm(dataloader):
        try:
            sentences = batch['sentence']
        except:
            sentences = batch['text']
        labels = batch['label']
        processed_data = []
        constituency_graphs = parser.get_graph(sentences)
        processed_data.append((constituency_graphs, labels))

        output_path = f"/usrvol/processed_data/{dataset}/{subset}/constituency" 
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        with open(f"{output_path}/{iterator}.pkl", 'wb') as f:
            pkl.dump(processed_data, f)
        iterator += 1


def get_graphs(dataset, subset, batch_size=8):
    constituency_parser = constituency_graph_generator(model='con-crf-roberta-en')
    build_graphs(dataset=dataset, subset=subset, parser=constituency_parser, batch_size=batch_size, )

datasets = ["SetFit/ag_news"]    # "stanfordnlp/sst2", 
subsets = ['train', 'test']

if __name__ == '__main__':
    
    for dataset in datasets:
        print(f"Processing {dataset} dataset...")
        for subset in subsets:
            print(f"Processing {subset} subset...")
            torch.cuda.empty_cache()
            folder = f"/usrvol/data/{dataset}/"            
            get_graphs(dataset=dataset, subset=subset, batch_size=256)    
            torch.cuda.empty_cache()