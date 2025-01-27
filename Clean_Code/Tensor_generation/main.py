import pickle as pkl
from transformers import AutoTokenizer, AutoModel, ElectraTokenizer
import os
from Embedding_generator_constituency import *
from tqdm import tqdm
from dataloader import *
torch.cuda.empty_cache()
folders = os.listdir('/usrvol/processed_data')

batch_size = 1

for folder in tqdm(folders):
    if folder in ['SNLI']: 
        print(f"Processing folder {folder}...")
        subfolders = os.listdir(f'/usrvol/processed_data/{folder}')
        for subfolder in subfolders:
            if subfolder in ['train','test','dev']:
                print(f"Processing subfolder {subfolder}...")
                graph_folders = os.listdir(f'/usrvol/processed_data/{folder}/{subfolder}')
                
                for model_name in tqdm(['bert-base-uncased'], colour='green'):
                    print(f"Generating tensors for {model_name}...")
                    new_graph_list = []
                    try:
                        output = model_name.split('/')[1]
                    except:
                        output = model_name
                    
                    if model_name in ['bert-base-uncased', 'albert/albert-base-v2']:
                        tokenizer = AutoTokenizer.from_pretrained(model_name)
                        model = AutoModel.from_pretrained(model_name)
                    elif model_name in ['microsoft/deberta-base']:
                        tokenizer = AutoTokenizer.from_pretrained(model_name)
                        model = AutoModel.from_pretrained(model_name)
                    elif model_name in ['google/electra-base-discriminator']:
                        tokenizer = ElectraTokenizer.from_pretrained(model_name)
                        model = AutoModel.from_pretrained(model_name)
                    model = model.to('cuda')
                    model.eval()
                    for graph_folder in graph_folders:
                        if graph_folder in ['semantic']:
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
                                
                                    for batch_idx, (graph1, graph2, labels) in enumerate(dataloader):

                                        sentences1 = get_words_from_graph(graph1)
                                        sentences2 = get_words_from_graph(graph2)
                                        sentence_tensors1 = get_tensors_for_sentence(sentences1, model, tokenizer, cuda=True)
                                        sentence_tensors2 = get_tensors_for_sentence(sentences2, model, tokenizer, cuda=True)
                                        data1 = torch_geometric_tensors(sentence_tensors1, graph1, model_name, mode=graph_folder)
                                        data2 = torch_geometric_tensors(sentence_tensors2, graph2, model_name, mode=graph_folder)
                                        assert len(data1.batch) == data1.num_nodes == len(data1.x)
                                        assert len(data2.batch) == data2.num_nodes == len(data2.x)
                                        assert len(data1.x) == data1.edge_index.max().item()+1
                                        assert len(data2.x) == data2.edge_index.max().item()+1
                                        new_graph_list.append((data1, data2, labels))

                                    output_folder = f"/usrvol/processed_tensors/{folder}/{subfolder}/{graph_folder}/{output}"
                                    if not os.path.exists(output_folder):
                                        os.makedirs(output_folder)
                                    with open(f"{output_folder}/{graph_name}.pkl", 'wb') as f:
                                        pkl.dump(new_graph_list, f)

                        torch.cuda.empty_cache()