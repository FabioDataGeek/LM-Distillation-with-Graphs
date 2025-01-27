import pickle as pkl
from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification
import os
from Embedding_generator_constituency import *
from tqdm import tqdm
from dataloader_constituency import *
from arguments import *

LABELS = {
    'SetFit/ag_news': 4,
    'stanfordnlp/sst2': 2
}

args = arguments()
folders = args['folders']
batch_size = args['batch_size']
model_name = args['model_name']


for dataset_name in args['datasets']:
    specific_tensors_route = f"/usrvol/utils/{dataset_name}_specific_tensors.pkl"
    folder = f"/usrvol/processed_data/{dataset_name}"
    print(f"Processing folder {folder}...")
    subfolders = os.listdir(f'{folder}')
    for subfolder in subfolders:
        print(f"Processing subfolder {subfolder}...")
        graph_folders = os.listdir(f'{folder}/{subfolder}') 
        print(f"Generating tensors for {model_name}...")
        new_graph_list = []
        model_route = f"/usrvol/results/{dataset_name}/best_fine_tuned_model.pt"
        model = AutoModelForSequenceClassification.from_pretrained(args['model_name'], num_labels=LABELS[dataset_name])
        state_dict = torch.load(model_route)
        model.load_state_dict(state_dict)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name)
        model = model.to('cuda')
        model.eval()
        for graph_folder in graph_folders:
            if graph_folder in args['mode']:
                print(f"Processing graph folder {graph_folder}...")
                graphs = os.listdir(f'{folder}/{subfolder}/{graph_folder}')
                for graph in tqdm(graphs, colour='red'):
                    if graph.endswith('.pkl'):
                        graph_name = graph.split('.')[0]
                        with open(f'{folder}/{subfolder}/{graph_folder}/{graph}', 'rb') as f:
                            graph_list = pkl.load(f)           
                        dataset = CustomDataset(graph_list)
                        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
                        new_graph_list = []
                    
                        for batch_idx, (specific_graph, labels) in enumerate(dataloader):
                            sentences = get_words_from_graph(specific_graph)
                            sentence_tensors = get_tensors_for_sentence(sentences, model, tokenizer)
                            data = torch_geometric_tensors(sentence_tensors,specific_tensors_route=specific_tensors_route, graphs=specific_graph, mode=graph_folder)
                            assert len(data.batch) == data.num_nodes == len(data.x)
                            assert len(data.x) == data.edge_index.max().item()+1
                            new_graph_list.append((data, labels))

                        output_folder = f"/usrvol/processed_tensors/{dataset_name}/{subfolder}/{graph_folder}"
                        if not os.path.exists(output_folder):
                            os.makedirs(output_folder)
                        with open(f"{output_folder}/{graph_name}.pkl", 'wb') as f:
                            pkl.dump(new_graph_list, f)

            torch.cuda.empty_cache()