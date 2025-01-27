from dicts import *
    
def arguments():
    mode = 'constituency'
    module = 'baseline'                  # GCNConv
    num_relations = 0
    if module in ['RGCNConv', 'RGATConv']:
        if mode == 'sintactic':
            num_relations = len(SINTACTIC_DICT)
        elif mode == 'semantic':
            num_relations = len(SEMANTIC_DICT)
        elif mode == 'constituency':
            num_relations = len(CONSTITUENCY_DICT)
        elif mode == 'sintactic+semantic':
            num_relations = len(SIN_SEM_DICT)
        elif mode == 'sintactic+constituency':
            num_relations = len(SIN_CON_DICT)
        elif mode == 'semantic+constituency':
            num_relations = len(SEM_CON_DICT)
        elif mode == 'sintactic+semantic+constituency':
            num_relations = len(SIN_SEM_CON_DICT)
    return {
        'num_epochs': 5,  # Number of epochs to train for
        'learning_rate': 1e-6,  # Learning rate for the optimizer   #2e-6
        'weight_decay': 1e-4,  # Weight decay for regularization
        'adam_epsilon': 1e-8,  # Epsilon value for the AdamW optimizer
        'root_train_data_path': f"/usrvol/processed_tensors/SNLI/train/{mode}/bert-base-uncased",  # Path to the training data
        'root_test_data_path': f"/usrvol/processed_tensors/SNLI/test/{mode}/bert-base-uncased",  # Path to the test data
        'root_dev_data_path': f"/usrvol/processed_tensors/SNLI/dev/{mode}/bert-base-uncased",  # Path to the development data
        'raw_train_data_path': f"/usrvol/processed_tensors/SNLI/train/{mode}/bert-base-uncased/raw",  # Path to the training data
        'raw_test_data_path': f"/usrvol/processed_tensors/SNLI/test/{mode}/bert-base-uncased/raw",  # Path to the test data
        'raw_dev_data_path': f"/usrvol/processed_tensors/SNLI/dev/{mode}/bert-base-uncased/raw",  # Path to the development data
        'size': 768,  # Size parameter for the model
        'num_layers': 1,  # Number of layers in the model
        'dropout': 0,  # Dropout rate  0.5 best for now
        'layer_norm': True,  # Whether to use layer normalization
        'cuda': True,  # Whether to use CUDA (GPU acceleration)
        'devices': 'cuda:0',  # Which device to use for CUDA
        'seed': 42,  # Seed for random number generation
        'fp16': False,  # Whether to use mixed precision training
        'loss_fn': 'CrossEntropyLoss',  # Loss function to use
        'lr_scheduler': 'fixed',  # Learning rate scheduler to use
        'module': module,  # GNN module to use or baseline
        'residual': False,  # Whether to use residual connections
        'batch_size': 2,  # Batch size for training
        'heads': 2,  # Number of attention heads
        'num_relations': num_relations+1,  # Number of relations in the graph
        'model_name': 'google-bert/bert-base-uncased',  # Name of the model to use
        'mode': mode,  # Mode of the model
        'pooling': 'max',  # Pooling method to use
        'lin_transform': None,  # Linear transformation size
        'undirected': True,  # Whether to use undirected edges
    }