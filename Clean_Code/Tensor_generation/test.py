import pickle as pkl
from transformers import AutoTokenizer, AutoModel, ElectraTokenizer
import os
from Embedding_generator import *
from tqdm import tqdm
from dataloader import *
torch.cuda.empty_cache()

with open('/usrvol/processed_data/SciTail/train/sintactic+semantic/sintactic_semantic46.pkl', 'rb') as f:
    graph_list = pkl.load(f)

#graph_list = graph_list[1:-1]
dataset = CustomDataset(graph_list)

""" for data in dataset:
    print(data) """
dataloader = DataLoader(dataset, batch_size=128, shuffle=False)
new_graph_list = []

for batch_idx, (graph1, graph2, labels) in enumerate(dataloader):
    print(batch_idx)

"""     sentences1 = get_words_from_graph(graph1)
    sentences2 = get_words_from_graph(graph2)
    sentence_tensors1 = get_tensors_for_sentence(sentences1, model, tokenizer)
    sentence_tensors2 = get_tensors_for_sentence(sentences2, model, tokenizer)
    
    data1 = torch_geometric_tensors(sentence_tensors1, graph1, model_name)
    data2 = torch_geometric_tensors(sentence_tensors2, graph2, model_name)
    new_graph_list.append((data1, data2, labels)) """