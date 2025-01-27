
""" Con este archivo, vamos a iterar sobre todas las oraciones de los datasets de entrenamiento para
determinar cual es el número máximo de tokens que se pueden encontrar en una oración. Con esto,
asignaremos un padding máximo a las oraciones para que todas tengan la misma longitud y si
es posible ahorraremos memoria si el máximo de tokens es inferior a 512. """

import pickle as pkl
from transformers import AutoTokenizer, AutoModel, ElectraTokenizer
import os
from Embedding_generator import *
from tqdm import tqdm
from dataloader import *
torch.cuda.empty_cache()
folders = os.listdir('/usrvol/processed_data')

batch_size = 1

max_tokens = 0
for folder in tqdm(folders):
    print(f"Processing folder {folder}...")
    subfolders = os.listdir(f'/usrvol/processed_data/{folder}')
    for subfolder in subfolders:
        print(f"Processing subfolder {subfolder}...")
        graph_folders = os.listdir(f'/usrvol/processed_data/{folder}/{subfolder}')
        for model_name in tqdm(['bert-base-uncased', 'albert/albert-base-v2', 'microsoft/deberta-base', 'google/electra-base-discriminator'], colour='green'):
            print(f"Generating tensors for {model_name}...")
            if model_name in ['bert-base-uncased', 'albert/albert-base-v2']:
                tokenizer = AutoTokenizer.from_pretrained(model_name)
            elif model_name in ['microsoft/deberta-base']:
                tokenizer = AutoTokenizer.from_pretrained(model_name)
            elif model_name in ['google/electra-base-discriminator']:
                tokenizer = ElectraTokenizer.from_pretrained(model_name)
            for graph_folder in graph_folders:
                if graph_folder in ['sintactic+semantic', 'sintactic+constituency', 'semantic+constituency', 'sintactic+semantic+constituency']:
                    continue
                else:
                    print(f"Processing graph folder {graph_folder}...")
                    graphs = os.listdir(f'/usrvol/processed_data/{folder}/{subfolder}/{graph_folder}')

                    for graph in tqdm(graphs, colour='red'):
                        if graph.endswith('.pkl'):
                            graph_name = graph.split('.')[0]
                            with open(f'/usrvol/processed_data/{folder}/{subfolder}/{graph_folder}/{graph}', 'rb') as f:
                                graph_list = pkl.load(f)           
                            dataset = CustomDataset(graph_list)
                            dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
                            new_graph_list = []
                            for data in dataloader:
                                graphs1 = data[0]
                                graphs2 = data[1]
                                labels = data[2]
                                sentences1 = get_words_from_graph(graphs1)
                                token_count = count_tokens(sentences1, tokenizer)
                                if token_count > max_tokens:
                                    max_tokens = token_count
                                sentences2 = get_words_from_graph(graphs2)
                                token_count = count_tokens(sentences2, tokenizer)
                                if token_count > max_tokens:
                                    max_tokens = token_count

print(f"Max tokens found in one sentence: {max_tokens}")
#150