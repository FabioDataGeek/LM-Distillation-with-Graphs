import json
from architecture_GNNs import *
from architecture_LLMs import *
from dataloader import *
from arguments import *
from torch.optim import AdamW
from tqdm import tqdm
import pickle as pkl
from datetime import datetime
from util import *
import time
import os
import shutil
from torch.utils.data import DataLoader as DataLoader_torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, AutoConfig
from datasets import load_dataset

all_data = {}
args = arguments()
generator = torch.Generator().manual_seed(args['seed'])

# MODEL
config = AutoConfig.from_pretrained(args['model_name'])
config.num_labels = 4
config.return_dict = True
model = AutoModelForSequenceClassification.from_pretrained(args['model_name'], config=config)
tokenizer = AutoTokenizer.from_pretrained(args['model_name'])
    
    
# DEVICE
if args['cuda']:
    device = torch.device(args['devices'] if torch.cuda.is_available() else "cpu")
    model.to(device)
    if not torch.cuda.is_available():
        print("WARNING: CUDA is not available, using CPU instead.")


# PARAMETERS
grouped_parameters = parameters(model=model, args=args)

# OPTIMIZER
optimizer = AdamW(params=grouped_parameters, weight_decay=args['weight_decay'], eps=args['adam_epsilon'])
scheduler = select_scheduler(optimizer=optimizer, lr_scheduler=args['lr_scheduler'])


# Count the number of trainable parameters in the model
print('parameters:')
for name, param in (model.named_parameters()):
    if param.requires_grad:
        print('\t{:45}\ttrainable\t{}\tdevice:{}'.format(name, param.size(), param.device))
    else:
        print('\t{:45}\tfixed\t{}\tdevice:{}'.format(name, param.size(), param.device))

num_params = sum(p.numel() for name, p in model.named_parameters() if p.requires_grad)
print('\ttotal:', num_params)


# DATASET
train_dataset = Dataset_LLM(file=args['root_train_data_path'], tokenizer=AutoTokenizer.from_pretrained(args['model_name']))
dev_dataset = Dataset_LLM(file=args['root_dev_data_path'], tokenizer=AutoTokenizer.from_pretrained(args['model_name']))
test_dataset = Dataset_LLM(file=args['root_test_data_path'], tokenizer=AutoTokenizer.from_pretrained(args['model_name']))
    
global_step = 0


# TRAINING, VALIDATION AND TEST IN EACH EPOCH
train_losses = []
val_losses = []
for i in tqdm(range(args['num_epochs'])):
    
    #Training
    print('Processing all the training files for epoch {}'.format(i))
    y_true = []
    y_pred = []
    
    dataloader = DataLoader_torch(dataset=train_dataset, batch_size=args['batch_size'], shuffle=True)
    y_true_list, y_pred_list, losses = train(model=model, train_loader=dataloader, 
        loss_fn=args['loss_fn'], optimizer=optimizer, 
        scheduler=scheduler, device=device, global_step=global_step, 
        fp16=args['fp16'])
    y_true.extend(y_true_list)
    y_pred.extend(y_pred_list)
    train_losses.extend(losses)
    # Report for each epoch
    dict_name = f"classification_report_train_epoch{i}"
    _ = reporting(y_true, y_pred, epoch=i, dict_name=dict_name)

    # Save losses
    with open('losses_train.json', 'w') as fp:
        json.dump(train_losses, fp)

    #Validation
    print('Processing all the validation files for epoch {}'.format(i))
    y_true = []
    y_pred = []

    dataloader = DataLoader_torch(dataset=dev_dataset, batch_size=args['batch_size'], shuffle=False)
    y_true_list, y_pred_list, losses = evaluation(model=model, dev_loader=dataloader,
        loss_fn=args['loss_fn'], device=device, fp16=args['fp16'])
    y_true.extend(y_true_list)
    y_pred.extend(y_pred_list)
        

    # Report and loss for each epoch
    dict_name = f"classification_report_dev_epoch{i}"
    _ = reporting(y_true, y_pred, epoch=i, dict_name=dict_name)
        
    #Testing
    print('Processing all the test files for epoch {}'.format(i))
    y_true = []
    y_pred = []

    dataloader = DataLoader_torch(dataset=test_dataset, batch_size=args['batch_size'], shuffle=False)
    y_true_list, y_pred_list = test(model=model, test_loader=dataloader, device=device, fp16=args['fp16'])
    y_true.extend(y_true_list)
    y_pred.extend(y_pred_list)

    # Report and loss for each epoch
    dict_name = f"classification_report_test_epoch{i}"
    _ = reporting(y_true, y_pred, epoch=i, dict_name=dict_name)


    # Save model
    """ torch.save(model.state_dict(), f"model{i}.pt") """
    start_time = time.time()

current_datetime = datetime.now().strftime('%Y-%m-%d_%H:%M:%S')
if not os.path.exists(f"results/{current_datetime}"):
    os.makedirs(f"results/{current_datetime}")  


with open(f"results/{current_datetime}/args.json", 'w') as f:
    json.dump(args, f)

# Move files to results folder
files = os.listdir()
for file in files:
    if file.endswith(".pt"):
        shutil.move(file, f"results/{current_datetime}/{file}")
    if file.endswith(".json"):
        shutil.move(file, f"results/{current_datetime}/{file}")

