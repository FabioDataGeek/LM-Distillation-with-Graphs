import torch
from tqdm import tqdm
import numpy as np
import json
from sklearn.metrics import classification_report
import time
from datetime import datetime
from transformers import get_constant_schedule, get_constant_schedule_with_warmup,  get_linear_schedule_with_warmup
import shutil
from arguments import *
import os

args = arguments()

LOSS_FUNCTIONS = {
    'CrossEntropyLoss': torch.nn.CrossEntropyLoss(),
    'BinaryCrossEntropyLoss': torch.nn.BCELoss(),
    'BCEWithLogitsLoss': torch.nn.BCEWithLogitsLoss()
}

def logs():
    current_datetime = datetime.now().strftime('%Y-%m-%d_%H:%M:%S')
    if not os.path.exists(f"results/{current_datetime}"):
        os.makedirs(f"results/{current_datetime}") 
    for i in range(len(args['epochs'])):
        shutil.move(f"model{i}.pt", f"results/{current_datetime}/model{i}.pt")
    shutil.move("results.json", f"results/{current_datetime}/results.json")
    shutil.move("losses_dict_train.json", f"results/{current_datetime}/losses_dict_train.json")
    shutil.move("losses_dict_dev.json", f"results/{current_datetime}/losses_dict_dev.json")
    shutil.move("classification_report_train.json", f"results/{current_datetime}/classification_report_train.json")
    shutil.move("classification_report_dev.json", f"results/{current_datetime}/classification_report_dev.json")
    shutil.move("classification_report_test.json", f"results/{current_datetime}/classification_report_test.json")

def parameters(model, args):
    if args['module'] in ['baseline', 'baseline-frozen']:
        grouped_parameters = [
            {'params': [p for n, p in model.named_parameters()], 'weight_decay': args['weight_decay'], 'lr': args['learning_rate']}
        ]
        return grouped_parameters
    else:
        grouped_parameters = [
            {'params': [p for n, p in model.gnn_layers.named_parameters()], 'weight_decay': args['weight_decay'], 'lr': args['learning_rate']},
            {'params': [p for n, p in model.mlp.named_parameters()], 'weight_decay': args['weight_decay'], 'lr': args['learning_rate']}
        ]
        return grouped_parameters
    

def calculate_warmup_steps(total_epochs, num_batches, warmup_proportion):
    total_steps = total_epochs * num_batches
    warmup_steps = int(total_steps * warmup_proportion)
    return warmup_steps

def select_scheduler(optimizer, lr_scheduler):
    if lr_scheduler == 'fixed':
        scheduler = get_constant_schedule(optimizer)
    else:
        raise NotImplementedError
    return scheduler

def reporting(y_true_list, y_pred_list, epoch, dict_name, dict={}):
    report = classification_report(y_true_list, y_pred_list, output_dict=True)
    name =dict_name.split('_')[2]
    name =name.upper()
    print(f"{name} CLASSIFICATION REPORT: ")
    print(classification_report(y_true_list, y_pred_list))
    dict = report
    with open(f"{dict_name}.json", 'w') as fp:
        json.dump(dict, fp)
    return report

def train(model, train_loader, loss_fn, optimizer, 
          scheduler, device, global_step, 
          fp16=False, scaler = None):
    
    # TRAINING
    if args['module'] in ['baseline-frozen']:
        for name, param in model.named_parameters():
            if 'classifier' in name:    
                param.requires_grad = True
            else:
                param.requires_grad = False
    else:
        for param in model.parameters():
            param.requires_grad = True
    log_interval = 10
    losses = []
    y_true_list = []
    y_pred_list = []
    model.train()
    start_time = time.time()
    # Usamos el train_loader para iterar sobre los datos de entrenamiento
    total_loss = 0 
    for batch in tqdm(train_loader):
        if args['module'] in ['baseline', 'baseline-frozen']:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label']
            labels = labels.to(device)
        else:            
            batch1, batch2, labels = batch
            batch1, batch2, labels = batch1.to(device), batch2.to(device), labels.to(device)
            labels = labels.squeeze()
        if args['module'] in ['GCNConv', 'GATConv', 'GraphConv', 'SAGEConv', 'GatedGraphConv', 'GINConv']:
            outputs = model(x1=batch1.x, x2=batch2.x, edge_index1=batch1.edge_index, edge_index2=batch2.edge_index, batch1=batch1.batch, batch2=batch2.batch)
            
            loss = LOSS_FUNCTIONS[loss_fn](outputs, labels)
            loss.backward()
            optimizer.step()
        if args['module'] in ['RGCNConv', 'RGATConv']:
            outputs = model(x1=batch1.x, x2=batch2.x, edge_index1=batch1.edge_index, edge_index2=batch2.edge_index, edge_type1=batch1.edge_attr, edge_type2=batch2.edge_attr, batch1=batch1.batch, batch2=batch2.batch)
            loss = LOSS_FUNCTIONS[loss_fn](outputs, labels)
            loss.backward()
            optimizer.step()
        if args['module'] in ['baseline', 'baseline-frozen']:
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            outputs = outputs[0]
            loss = LOSS_FUNCTIONS[loss_fn](outputs, labels)
            loss.backward()
            optimizer.step()
        
        scheduler.step()

        # Append the loss to the list of losses
        losses.append(loss.item())
        total_loss = loss.item()
        y_true_list.extend(labels.detach().cpu().numpy())
        outputs = outputs.detach().cpu().numpy()
        outputs = np.argmax(outputs, axis=1)
        y_pred_list.extend(list(outputs))

        """ # Print info
        ms_per_batch = 1000 * (time.time() - start_time) / log_interval
        print('|step {:5} |loss {:7.4f} |ms/batch {:7.2f}|'.format(global_step, total_loss, ms_per_batch))
        
        start_time = time.time()
        global_step += 1 """

    return y_true_list, y_pred_list, losses
    

def evaluation(model, dev_loader, loss_fn, device, fp16=False):
    model.eval()
    with torch.no_grad():
        # Usamos el dev_loader para iterar sobre los datos de validación
        losses = []
        y_true_list = []
        y_pred_list = []
        total_loss = 0
        for batch in tqdm(dev_loader):
            if args['module'] in ['baseline', 'baseline-frozen']:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['label']
                labels = labels.to(device)
            else:
            # Get data and labels from batch
                batch1, batch2, labels = batch
                batch1, batch2, labels = batch1.to(device), batch2.to(device), labels.to(device)
                labels = labels.squeeze()
            if args['module'] in ['GCNConv', 'GATConv', 'GraphConv', 'SAGEConv', 'GatedGraphConv', 'GINConv']:
                outputs = model(x1=batch1.x, x2=batch2.x, edge_index1=batch1.edge_index, edge_index2=batch2.edge_index, batch1=batch1.batch, batch2=batch2.batch)
                loss = LOSS_FUNCTIONS[loss_fn](outputs, labels)
            if args['module'] in ['RGCNConv', 'RGATConv']:
                outputs = model(x1=batch1.x, x2=batch2.x, edge_index1=batch1.edge_index, edge_index2=batch2.edge_index, edge_type1=batch1.edge_attr, edge_type2=batch2.edge_attr, batch1=batch1.batch, batch2=batch2.batch)
                loss = LOSS_FUNCTIONS[loss_fn](outputs, labels)
            if args['module'] in ['baseline', 'baseline-frozen']:
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                outputs = outputs[0]
                loss = LOSS_FUNCTIONS[loss_fn](outputs, labels)


            # Calculate loss
            total_loss += loss.item()
            
            # Classification Report
            y_true_list.extend(labels.detach().cpu().numpy())
            outputs = outputs.detach().cpu().numpy()
            outputs = np.argmax(outputs, axis=1)
            y_pred_list.extend(list(outputs))
            losses.append(total_loss)
    
    return y_true_list, y_pred_list, losses

def test(model, test_loader, device, fp16=False):
    # TESTING
    model.eval()
    # Usamos el test_loader para iterar sobre los datos de test
    y_true_list = []
    y_pred_list = []
    with torch.no_grad():
        for batch in tqdm(test_loader):
            if args['module'] in ['baseline', 'baseline-frozen']:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['label']
                labels = labels.to(device)
            else:
            # Get data and labels from batch
                batch1, batch2, labels = batch
                batch1, batch2, labels = batch1.to(device), batch2.to(device), labels.to(device)
                labels = labels.squeeze()
            if args['module'] in ['GCNConv', 'GATConv', 'GraphConv', 'SAGEConv', 'GatedGraphConv', 'GINConv']:
                outputs = model(x1=batch1.x, x2=batch2.x, edge_index1=batch1.edge_index, edge_index2=batch2.edge_index, batch1=batch1.batch, batch2=batch2.batch)
            if args['module'] in ['RGCNConv', 'RGATConv']:
                outputs = model(x1=batch1.x, x2=batch2.x, edge_index1=batch1.edge_index, edge_index2=batch2.edge_index, edge_type1=batch1.edge_attr, edge_type2=batch2.edge_attr, batch1=batch1.batch, batch2=batch2.batch)
            if args['module'] in ['baseline', 'baseline-frozen']:
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                outputs = outputs[0]
            
            # Classification Report
            y_true_list.extend(labels.detach().cpu().numpy())
            outputs = outputs.detach().cpu().numpy()
            outputs = np.argmax(outputs, axis=1)
            y_pred_list.extend(list(outputs))

    return y_true_list, y_pred_list