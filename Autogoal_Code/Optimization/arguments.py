from dicts import *

DATASET_LENGTH = {
    'SetFit/ag_news': 7600,
    'stanfordnlp/sst2': 872,
}    
    
def arguments():
    dataset = 'stanfordnlp/sst2'
    DATASET_NAME = { 
                    'SetFit/ag_news': 'ag-news',
                    'stanfordnlp/sst2': 'sst2'
                    }
    LABELS = {
    'SetFit/ag_news': 4,
    'stanfordnlp/sst2': 2
    }
    
    module =  'GCNConv'
    mode = 'constituency'
    return {
        'dataset_length': DATASET_LENGTH[dataset],  # Number of samples in the dataset
        'root_test_data_path': f"/home/coder/autogoal/data/{DATASET_NAME[dataset]}",  # Path to the test data
        'raw_test_data_path': f"/home/coder/autogoal/data/{DATASET_NAME[dataset]}/raw",  # Path to the test data
        'size': 768,  # Size parameter for the model
        'seed': 42,
        'cuda': True,
        'devices': 'cuda:0',
        'num_layers': 3,  # Number of layers in the model
        'dropout': 0.2,  # Dropout rate
        'layer_norm': True,  # Whether to use layer normalization
        'module': module,  # GNN module to use or baseline
        'residual': False,  # Whether to use residual connections
        'batch_size': 1,  # Batch size for training
        'mode': mode,  # Mode of the model
        'pooling': 'max',  # Pooling method to use
        'lin_transform': None,  # Linear transformation size
        'labels' : LABELS[dataset],
        'dataset': dataset,
        'undirected': True,  # Whether to use undirected edges
        'model_dir': f"/home/coder/autogoal/model/{DATASET_NAME[dataset]}_GNN.pt",
        'sampling': 'stratified',                                        # [stratified, weighted, maximal_diversity, secondary_clustering_sampling]
        'n_subclusters': 4,                                             # Number of subclusters to sample from each cluster
        'number_of_clusters': 3,                                        # 3                                          # Number of clusters to sample from the dataset
        'num_samples': 100,                                              # Number of samples to draw from each cluster
    }