import json
import os
import shutil
import time
from datetime import datetime
from tqdm import tqdm
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader as DataLoader_torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, AutoConfig
from architecture_GNNs import GNN_classifier, RGNN_classifier
from dataloader import Dataset_GNN_2graphs, Dataset_RGNN_2graphs, HomogeneousDataLoader_2graphs, HeteroGeneousDataLoader_2graphs
from arguments import arguments
from util import parameters, select_scheduler, train, evaluation, test, reporting
import torch_geometric

# Set high precision for matrix multiplication
torch.set_float32_matmul_precision('high')

# Initialize arguments and seed
args = arguments()
generator = torch.Generator().manual_seed(args['seed'])

# Model Selection
if args['module'] in ['GCNConv', 'GATConv', 'GraphConv', 'SAGEConv']:
    model = GNN_classifier(size=args['size'], num_layers=args['num_layers'], dropout=args['dropout'], 
                           module=args['module'], layer_norm=args['layer_norm'], residual=args['residual'], 
                           pooling='max', lin_transform=args['lin_transform'])
elif args['module'] in ['RGCNConv', 'RGATConv']:
    model = RGNN_classifier(size=args['size'], num_layers=args['num_layers'], dropout=args['dropout'], 
                            module=args['module'], layer_norm=args['layer_norm'], residual=args['residual'], 
                            num_relations=args['num_relations'], pooling='max')
else:
    raise ValueError(f"Unsupported module type: {args['module']}")

# Set device
device = torch.device(args['devices'] if args['cuda'] and torch.cuda.is_available() else "cpu")
model.to(device)
if args['cuda'] and not torch.cuda.is_available():
    print("WARNING: CUDA is not available, using CPU instead.")

# Set parameters, optimizer, and scheduler
params = parameters(model=model, args=args)
optimizer = AdamW(params=params, weight_decay=args['weight_decay'], eps=args['adam_epsilon'])
scheduler = select_scheduler(optimizer=optimizer, lr_scheduler=args['lr_scheduler'])

# Count the number of trainable parameters
num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f'Total trainable parameters: {num_params}')

# Load datasets
if args['module'] in ['GCNConv', 'GATConv', 'GraphConv', 'SAGEConv']:
    train_dataset = Dataset_GNN_2graphs(root=args['root_train_data_path'], files_path=args['raw_train_data_path'])
    dev_dataset = Dataset_GNN_2graphs(root=args['root_dev_data_path'], files_path=args['raw_dev_data_path'])
    test_dataset = Dataset_GNN_2graphs(root=args['root_test_data_path'], files_path=args['raw_test_data_path'])
elif args['module'] in ['RGCNConv', 'RGATConv']:
    mode_flags = {'semantic': False, 'sintactic': False, 'constituency': False}
    for m in args['mode'].split('+'):
        if m in mode_flags:
            mode_flags[m] = True
    train_dataset = Dataset_RGNN_2graphs(root=args['root_train_data_path'], files_path=args['raw_train_data_path'], **mode_flags)
    dev_dataset = Dataset_RGNN_2graphs(root=args['root_dev_data_path'], files_path=args['raw_dev_data_path'], **mode_flags)
    test_dataset = Dataset_RGNN_2graphs(root=args['root_test_data_path'], files_path=args['raw_test_data_path'], **mode_flags)
else:
    raise ValueError(f"Unsupported module type for datasets: {args['module']}")

# Training, validation, and testing loop
train_losses = []
val_losses = []
global_step = 0

for epoch in range(args['num_epochs']):
    print(f'Starting epoch {epoch}')

    # Training
    model.train()
    y_true, y_pred = [], []
    for file_idx in tqdm(range(len(train_dataset)), desc=f"Training Epoch {epoch}"):
        dataloader = (HomogeneousDataLoader_2graphs(train_dataset[file_idx], batch_size=args['batch_size'], shuffle=True) 
                      if args['module'] in ['GCNConv', 'GATConv', 'GraphConv', 'SAGEConv'] 
                      else HeteroGeneousDataLoader_2graphs(train_dataset[file_idx], batch_size=args['batch_size'], shuffle=True))
        y_true_list, y_pred_list, losses = train(model=model, train_loader=dataloader, loss_fn=args['loss_fn'], 
                                                 optimizer=optimizer, scheduler=scheduler, device=device, 
                                                 global_step=global_step, fp16=args['fp16'])
        y_true.extend(y_true_list)
        y_pred.extend(y_pred_list)
        train_losses.extend(losses)
    reporting(y_true, y_pred, epoch=epoch, dict_name=f"classification_report_train_epoch{epoch}")

    # Save training losses
    with open('losses_train.json', 'w') as fp:
        json.dump(train_losses, fp)

    # Validation
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for file_idx in tqdm(range(len(dev_dataset)), desc=f"Validation Epoch {epoch}"):
            dataloader = (HomogeneousDataLoader_2graphs(dev_dataset[file_idx], batch_size=args['batch_size'], shuffle=False) 
                          if args['module'] in ['GCNConv', 'GATConv', 'GraphConv', 'SAGEConv'] 
                          else HeteroGeneousDataLoader_2graphs(dev_dataset[file_idx], batch_size=args['batch_size'], shuffle=False))
            y_true_list, y_pred_list, _ = evaluation(model=model, dev_loader=dataloader, loss_fn=args['loss_fn'], 
                                                     device=device, fp16=args['fp16'])
            y_true.extend(y_true_list)
            y_pred.extend(y_pred_list)
    reporting(y_true, y_pred, epoch=epoch, dict_name=f"classification_report_dev_epoch{epoch}")

    # Testing
    y_true, y_pred = [], []
    with torch.no_grad():
        for file_idx in tqdm(range(len(test_dataset)), desc=f"Testing Epoch {epoch}"):
            dataloader = (HomogeneousDataLoader_2graphs(test_dataset[file_idx], batch_size=args['batch_size'], shuffle=False) 
                          if args['module'] in ['GCNConv', 'GATConv', 'GraphConv', 'SAGEConv'] 
                          else HeteroGeneousDataLoader_2graphs(test_dataset[file_idx], batch_size=args['batch_size'], shuffle=False))
            y_true_list, y_pred_list = test(model=model, test_loader=dataloader, device=device, fp16=args['fp16'])
            y_true.extend(y_true_list)
            y_pred.extend(y_pred_list)
    reporting(y_true, y_pred, epoch=epoch, dict_name=f"classification_report_test_epoch{epoch}")

    # Save model
    torch.save(model.state_dict(), f"model_epoch_{epoch}.pt")

# Save results
current_datetime = datetime.now().strftime('%Y-%m-%d_%H:%M:%S')
results_dir = f"results/{current_datetime}"
os.makedirs(results_dir, exist_ok=True)

# Save arguments and move related files to results folder
with open(f"{results_dir}/args.json", 'w') as f:
    json.dump(vars(args), f)

for file in os.listdir():
    if file.endswith(".pt") or file.endswith(".json"):
        shutil.move(file, f"{results_dir}/{file}")
