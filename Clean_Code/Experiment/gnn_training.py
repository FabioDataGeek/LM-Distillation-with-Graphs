import json
import os
import shutil
import time
from datetime import datetime
from tqdm import tqdm
import torch
from torch.optim import AdamW
from torch.nn import CrossEntropyLoss
import numpy as np
from architecture_GNNs_single import GNN_classifier
from dataloader import Dataset_GNN_guided, HomogeneousDataLoader
from arguments_gnn import arguments
from util import parameters, select_scheduler, reporting
import random

# Set high precision for matrix multiplication
torch.set_float32_matmul_precision('high')

# Initialize arguments and seed
args = arguments()
torch.manual_seed(args['seed'])  # Set the seed for reproducibility in PyTorch
random.seed(args['seed'])  # Set the seed for reproducibility in random operations

# Model Selection
def initialize_model(args):
    # Initialize the model based on the specified module type
    if args['module'] in ['GCNConv', 'GATConv', 'GraphConv', 'SAGEConv']:
        return GNN_classifier(size=args['size'], num_layers=args['num_layers'], dropout=args['dropout'], 
                              module=args['module'], layer_norm=args['layer_norm'], residual=args['residual'], 
                              pooling='max', lin_transform=args['lin_transform'])
    else:
        raise ValueError(f"Unsupported module type: {args['module']}")

# Set device
def get_device(args):
    # Set the device to GPU if available and CUDA is enabled, otherwise use CPU
    if args['cuda'] and torch.cuda.is_available():
        return torch.device(args['devices'])
    else:
        if args['cuda'] and not torch.cuda.is_available():
            print("WARNING: CUDA is not available, using CPU instead.")
        return torch.device("cpu")

# Training, Validation, and Testing Loops
def run_epoch(model, dataset, device, optimizer=None, scheduler=None, training=False):
    dataloader_fn = HomogeneousDataLoader
    loss_fn = CrossEntropyLoss()  # Use CrossEntropyLoss for classification tasks
    model.train() if training else model.eval()  # Set model to train or eval mode based on the phase
    y_true, y_pred, losses = [], [], []

    # Shuffle file indices during training for better generalization
    if training:
        file_indices = list(range(len(dataset)))
        random.shuffle(file_indices)
    else:
        file_indices = range(len(dataset))

    # Iterate over each file in the dataset
    for file_idx in tqdm(file_indices, desc=("Training" if training else "Evaluating")):
        dataloader = dataloader_fn(dataset[file_idx], batch_size=args['batch_size'], shuffle=training)
        
        # Iterate over each batch in the dataloader
        for batch in tqdm(dataloader, desc=f"{'Training' if training else 'Evaluating'} File {file_idx}"):
            data, labels = batch
            # Move data and labels to the specified device (CPU or GPU)
            data, labels = data.to(device), labels.squeeze().to(device)
            
            # Extract relevant information from the data
            x, edge_index, batch = data.x, data.edge_index, data.batch

            # Forward pass
            outputs = model(x=x, edge_index=edge_index, batch=batch)
            loss = loss_fn(outputs, labels)  # Compute the loss
            
            if training:
                optimizer.zero_grad()  # Zero the gradients before backward pass
                loss.backward()  # Backward pass
                optimizer.step()  # Update model parameters
                if scheduler:
                    scheduler.step()  # Update learning rate if using a scheduler
            
            losses.append(loss.item())  # Store the loss for tracking
            y_true.extend(labels.cpu().numpy())  # Store the true labels
            y_pred.extend(torch.argmax(outputs, dim=1).detach().cpu().numpy())  # Store the predicted labels
    
    return y_true, y_pred, losses  # Return true labels, predicted labels, and losses for reporting

# Main Training Script
def main():
    # Initialize model, device, optimizer, and scheduler
    model = initialize_model(args)
    device = get_device(args)
    model.to(device)  # Move model to the specified device

    # Set up parameters, optimizer, and learning rate scheduler
    params = parameters(model=model, args=args)
    optimizer = AdamW(params=params, weight_decay=args['weight_decay'], eps=args['adam_epsilon'])
    scheduler = select_scheduler(optimizer=optimizer, lr_scheduler=args['lr_scheduler'])

    # Load datasets for training, validation, and testing
    train_dataset = Dataset_GNN_guided(root=args['root_train_data_path'], files_path=args['raw_train_data_path'])
    if args['dataset'] != 'SetFit/ag_news':
        dev_dataset = Dataset_GNN_guided(root=args['root_dev_data_path'], files_path=args['raw_dev_data_path'])
    test_dataset = Dataset_GNN_guided(root=args['root_test_data_path'], files_path=args['raw_test_data_path'])

    # Training Loop
    train_losses = []
    for epoch in range(args['num_epochs']):
        print(f'Starting epoch {epoch}')

        # Training Phase
        y_true, y_pred, losses = run_epoch(model, train_dataset, device, optimizer, scheduler, training=True)
        train_losses.extend(losses)  # Accumulate training losses
        reporting(y_true, y_pred, epoch=epoch, dict_name=f"classification_report_train_epoch{epoch}")  # Generate training report

        # Validation Phase
        if args['dataset'] != 'SetFit/ag_news':
            y_true, y_pred, _ = run_epoch(model, dev_dataset, device, training=False)
            reporting(y_true, y_pred, epoch=epoch, dict_name=f"classification_report_dev_epoch{epoch}")  # Generate validation report

        # Testing Phase
        y_true, y_pred, _ = run_epoch(model, test_dataset, device, training=False)
        reporting(y_true, y_pred, epoch=epoch, dict_name=f"classification_report_test_epoch{epoch}")  # Generate testing report

        # Save model checkpoint for the current epoch
        torch.save(model.state_dict(), f"model_epoch_{epoch}.pt")

    # Save training losses at the end
    with open('losses_train.json', 'w') as fp:
        json.dump(train_losses, fp)

    # Save results directory with current datetime to make it unique
    current_datetime = datetime.now().strftime('%Y-%m-%d_%H:%M:%S')
    results_dir = f"results/{current_datetime}"
    os.makedirs(results_dir, exist_ok=True)

    # Save arguments and move related files to results folder
    with open(f"{results_dir}/args.json", 'w') as f:
        json.dump(args, f)

    # Move model and loss files to the results directory for better organization
    for file in os.listdir():
        if file.endswith(".pt") or file.endswith(".json"):
            shutil.move(file, f"{results_dir}/{file}")

if __name__ == "__main__":
    main()
