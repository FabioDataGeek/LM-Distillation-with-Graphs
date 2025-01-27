import json
import os
import shutil
import torch
import numpy as np
from datetime import datetime
from tqdm import tqdm
from torch.optim import AdamW
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer, AutoConfig, get_linear_schedule_with_warmup
from datasets import load_dataset
from arguments import arguments
from util import *
import logging
from torch.cuda.amp import GradScaler, autocast


def convert_to_native(obj):
    if isinstance(obj, dict):
        return {k: convert_to_native(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_native(i) for i in obj]
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()  # Convert arrays to lists
    else:
        return obj


# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Set of datasets to work with
datasets_to_use = ["stanfordnlp/sst2", "SetFit/ag_news"]

# Initial setup
args = arguments()
generator = torch.Generator().manual_seed(args['seed'])
global_step = 0

# Iterate through the list of datasets
for dataset_name in datasets_to_use:
    logging.info(f"Loading dataset: {dataset_name}")
    try:
        # Load the dataset
        dataset = load_dataset(dataset_name)
        train_dataset = dataset['train']
    except KeyError:
        # Handle the case where the train dataset is not found
        logging.warning(f"Train dataset not found for {dataset_name}. Skipping this dataset.")
        continue

    dev_dataset = dataset.get('validation', None)
    test_dataset = dataset.get('test', None)

    # Determine the number of labels by checking the unique values in the 'label' column
    if 'label' in train_dataset.column_names:
        num_labels = len(set(train_dataset['label']))
    else:
        raise ValueError(f"The dataset {dataset_name} does not contain a 'label' column.")

    # MODEL
    # Load model configuration and set the number of labels dynamically based on the dataset
    config = AutoConfig.from_pretrained(args['model_name'])
    config.num_labels = num_labels
    config.return_dict = True
    model = AutoModelForSequenceClassification.from_pretrained(args['model_name'], config=config)
    tokenizer = AutoTokenizer.from_pretrained(args['model_name'])

    # DEVICE
    # Move the model to GPU if available, otherwise use CPU
    device = torch.device('cuda' if args['cuda'] and torch.cuda.is_available() else 'cpu')
    model.to(device)

    # PARAMETERS
    # Group model parameters for the optimizer
    grouped_parameters = parameters(model=model, args=args)

    # OPTIMIZER AND SCHEDULER
    # Set up the optimizer and learning rate scheduler
    optimizer = AdamW(params=grouped_parameters)
    total_steps = len(train_dataset) // args['batch_size'] * args['num_epochs']
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

    # Count the number of trainable parameters in the model
    logging.info('parameters:')
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logging.info(f'Total trainable parameters: {num_params}')

    # Adding an index to keep track of original positions in the dataset
    train_dataset = train_dataset.map(lambda example, idx: {'index': idx}, with_indices=True)
    if dev_dataset is not None:
        dev_dataset = dev_dataset.map(lambda example, idx: {'index': idx}, with_indices=True)
    if test_dataset is not None:
        test_dataset = test_dataset.map(lambda example, idx: {'index': idx}, with_indices=True)

    # Tokenizing datasets
    def tokenize_function(examples):
        # Use 'sentence' or 'text' as the key for the input text
        text_key = 'sentence' if 'sentence' in examples else 'text'
        tokenized_output = tokenizer(examples[text_key], truncation=True, padding='max_length', max_length=512)
        tokenized_output['index'] = examples['index']  # Keep the original index for reference
        if 'label' in examples:
            tokenized_output['label'] = examples['label']
        return tokenized_output

    # Apply tokenization to all datasets and remove original columns
    train_dataset = train_dataset.map(tokenize_function, batched=True, remove_columns=train_dataset.column_names)
    if dev_dataset is not None:
        dev_dataset = dev_dataset.map(tokenize_function, batched=True, remove_columns=dev_dataset.column_names)
    if test_dataset is not None:
        test_dataset = test_dataset.map(tokenize_function, batched=True, remove_columns=test_dataset.column_names)

    # Set format for datasets
    format_columns = ['input_ids', 'attention_mask', 'label', 'index']
    available_columns = [col for col in format_columns if col in train_dataset.column_names]
    train_dataset.set_format(type='torch', columns=available_columns)
    if dev_dataset is not None:
        available_columns = [col for col in format_columns if col in dev_dataset.column_names]
        dev_dataset.set_format(type='torch', columns=available_columns)
    if test_dataset is not None:
        available_columns = [col for col in format_columns if col in test_dataset.column_names]
        test_dataset.set_format(type='torch', columns=available_columns)

    # Create dataloaders for train, validation, and test datasets
    train_dataloader = DataLoader(dataset=train_dataset, batch_size=args['batch_size'], shuffle=True, pin_memory=True)
    dev_dataloader = DataLoader(dataset=dev_dataset, batch_size=args['batch_size'], shuffle=False, pin_memory=True) if dev_dataset is not None else None
    test_dataloader = DataLoader(dataset=test_dataset, batch_size=args['batch_size'], shuffle=False, pin_memory=True) if test_dataset is not None else None

    # Set up FP16 training using GradScaler
    scaler = GradScaler()

    # TRAINING, VALIDATION AND TEST IN EACH EPOCH
    results_dir = f"results/{dataset_name}"
    os.makedirs(results_dir, exist_ok=True)

    all_predictions = []
    aggregated_losses = {'train': [], 'validation': []}

    # Loop through each epoch for training, validation, and testing
    for epoch in range(args['num_epochs']):
        model.train()
        logging.info(f'Starting training for epoch {epoch}')
        train_losses = []
        y_true, y_pred, train_data_references = [], [], []

        # Training loop
        for batch in tqdm(train_dataloader, total=len(train_dataloader)):
            input_ids = batch['input_ids'].to(device, non_blocking=True)
            attention_mask = batch['attention_mask'].to(device, non_blocking=True)
            labels = batch['label'].to(device, non_blocking=True)
            indices = batch['index'].tolist()

            # Forward pass with FP16
            with autocast():
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs.logits
                loss = CrossEntropyLoss()(logits, labels)

            # Backward pass with FP16 scaling
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            # Apply gradient clipping to stabilize training
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scheduler.step()  # Scheduler step should be called after optimizer step
            scaler.update()

            # Track loss and predictions
            train_losses.append(loss.item())
            y_true.extend(labels.cpu().numpy())
            preds = torch.argmax(logits, axis=1).cpu().numpy()
            y_pred.extend(preds)
            train_data_references.extend(indices)

        # Save predictions and references for the training epoch
        for true_label, pred_label, data_ref in zip(y_true, y_pred, train_data_references):
            all_predictions.append({
                'epoch': epoch,
                'dataset': 'train',
                'true_label': true_label,
                'predicted_label': pred_label,
                'data_index': data_ref
            })

        # Generate report for training epoch
        dict_name = f"classification_report_train_epoch{epoch}"
        reporting(y_true, y_pred, epoch=epoch, dict_name=dict_name)

        # Save training losses to aggregated list
        aggregated_losses['train'].extend(train_losses)

        if dev_dataloader is not None:
            # Validation
            logging.info(f'Starting validation for epoch {epoch}')
            model.eval()
            y_true, y_pred, dev_data_references = [], [], []
            val_losses = []

            # Validation loop
            with torch.no_grad():
                for batch in tqdm(dev_dataloader, total=len(dev_dataloader)):
                    input_ids = batch['input_ids'].to(device, non_blocking=True)
                    attention_mask = batch['attention_mask'].to(device, non_blocking=True)
                    labels = batch['label'].to(device, non_blocking=True)
                    indices = batch['index'].tolist()

                    # Forward pass with FP16
                    with autocast():
                        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                        logits = outputs.logits
                        loss = CrossEntropyLoss()(logits, labels)

                    val_losses.append(loss.item())
                    y_true.extend(labels.cpu().numpy())
                    preds = torch.argmax(logits, axis=1).cpu().numpy()
                    y_pred.extend(preds)
                    dev_data_references.extend(indices)

            # Save predictions and references for validation
            for true_label, pred_label, data_ref in zip(y_true, y_pred, dev_data_references):
                all_predictions.append({
                    'epoch': epoch,
                    'dataset': 'validation',
                    'true_label': true_label,
                    'predicted_label': pred_label,
                    'data_index': data_ref
                })

            # Generate report for validation
            dict_name = f"classification_report_dev_epoch{epoch}"
            reporting(y_true, y_pred, epoch=epoch, dict_name=dict_name)

            # Save validation losses to aggregated list
            aggregated_losses['validation'].extend(val_losses)

        if test_dataloader is not None:
            # Testing
            logging.info(f'Starting testing for epoch {epoch}')
            model.eval()
            y_true, y_pred, test_data_references = [], [], []
            test_losses = []

            # Testing loop
            with torch.no_grad():
                for batch in tqdm(test_dataloader, total=len(test_dataloader)):
                    input_ids = batch['input_ids'].to(device, non_blocking=True)
                    attention_mask = batch['attention_mask'].to(device, non_blocking=True)
                    labels = batch['label'].to(device, non_blocking=True)
                    indices = batch['index'].tolist()

                    # Forward pass with FP16
                    with autocast():
                        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                        logits = outputs.logits

                    y_true.extend(labels.cpu().numpy())
                    preds = torch.argmax(logits, axis=1).cpu().numpy()
                    y_pred.extend(preds)
                    test_data_references.extend(indices)

            # Save predictions and references for testing
            for true_label, pred_label, data_ref in zip(y_true, y_pred, test_data_references):
                all_predictions.append({
                    'epoch': epoch,
                    'dataset': 'test',
                    'true_label': true_label,
                    'predicted_label': pred_label,
                    'data_index': data_ref
                })

            # Generate report for testing
            dict_name = f"classification_report_test_epoch{epoch}"
            reporting(y_true, y_pred, epoch=epoch, dict_name=dict_name)

            # Save test losses to aggregated list
            aggregated_losses['test'] = test_losses

        # Save model checkpoint
        torch.save(model.state_dict(), f'{results_dir}/model_epoch_{epoch}.pt')

        # Save aggregated losses
        with open(f'{results_dir}/aggregated_losses_{dataset_name.replace("/", "_")}.json', 'w') as fp:
            json.dump(aggregated_losses, fp)

        # Save arguments
        with open(os.path.join(results_dir, f'args_{dataset_name.replace("/", "_")}.json'), 'w') as f:
            json.dump(args, f)

        # Save all predictions
        all_predictions_native = convert_to_native(all_predictions)
        with open(os.path.join(results_dir, f'predictions.json'), 'w') as f:
            json.dump(all_predictions_native, f)

    # Move related files to results folder
    files = os.listdir()
    for file in files:
        if (file.endswith(".pt") or file.endswith(".json")) and not file.startswith("results"):
            shutil.move(file, results_dir)