from dicts import *
    

LABELS = {
    'SetFit/ag_news': 4,
    'stanfordnlp/sst2': 2
}
def arguments():
    dataset = 'stanfordnlp/sst2'
    module = 'GCNConv'                  # GCNConv
    num_relations = 0
    if module in ['RGCNConv', 'RGATConv']:
        num_relations = len(CONSTITUENCY_DICT)
    return {
        'num_epochs': 10,  # Number of epochs to train for
        'learning_rate': 1e-6,  # Learning rate for the optimizer   #2e-6
        'weight_decay': 1e-4,  # Weight decay for regularization
        'adam_epsilon': 1e-8,  # Epsilon value for the AdamW optimizer
        'root_train_data_path': f"/usrvol/processed_tensors/{dataset}/train_with_lm_labels",  # Path to the training data
        'root_test_data_path': f"/usrvol/processed_tensors/{dataset}/test_with_lm_labels",  # Path to the test data
        'root_dev_data_path': f"/usrvol/processed_tensors/{dataset}/validation_with_lm_labels",  # Path to the development data
        'raw_train_data_path': f"/usrvol/processed_tensors/{dataset}/train_with_lm_labels/raw",  # Path to the training data
        'raw_test_data_path': f"/usrvol/processed_tensors/{dataset}/test_with_lm_labels/raw",  # Path to the test data
        'raw_dev_data_path': f"/usrvol/processed_tensors/{dataset}/validation_with_lm_labels/raw",  # Path to the development data
        'size': 768,  # Size parameter for the model
        'num_layers': 3,  # Number of layers in the model
        'dropout': 0.2,  # Dropout rate for the model
        'layer_norm': True,  # Whether to use layer normalization
        'cuda': True,  # Whether to use CUDA (GPU acceleration)
        'devices': 'cuda:0',  # Which device to use for CUDA
        'seed': 42,  # Seed for random number generation
        'fp16': False,  # Whether to use mixed precision training
        'loss_fn': 'CrossEntropyLoss',  # Loss function to use
        'lr_scheduler': 'fixed',  # Learning rate scheduler to use
        'module': module,  # GNN module to use or baseline
        'residual': False,  # Whether to use residual connections
        'batch_size': 20,  # Batch size for training
        'heads': 2,  # Number of attention heads
        'num_relations': num_relations+1,  # Number of relations in the graph
        'pooling': 'max',  # Pooling method to use
        'lin_transform': None,  # Linear transformation size
        'undirected': True,  # Whether to use undirected edges
        'labels' : LABELS[dataset],
        'dataset': dataset
    }