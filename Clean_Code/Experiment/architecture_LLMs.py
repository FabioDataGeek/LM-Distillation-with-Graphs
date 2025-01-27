import transformers
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
from transformers import BertForSequenceClassification, AutoTokenizer, AutoConfig

""" Configurar las clases para que los pesos del modelo se puedan descongelar o no según un parámetro """



# Define the label mappings
num_labels = 4  
label2id = {'contradiction': 0, 'entailment': 1, 'neutral': 2, '-': 3}
id2label = {0: 'contradiction', 1: 'entailment', 2: 'neutral', 3: '-'}


""" class Concat_Classifier(nn.Module):
    def __init__(self, model_name):
        super(Concat_Classifier, self).__init__()
        self.model_name = model_name   #"google-bert/bert-base-uncased"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.config = AutoConfig.from_pretrained(model_name)
        self.config.num_labels = num_labels
        self.model =   BertForSequenceClassification.from_pretrained(self.model_name, config=self.config)    
    
    def forward(self, input_ids, attention_mask, token_type_ids, labels):
        #input_ids, attention_mask, token_type_ids = inputs['input_ids'], inputs['attention_mask'], inputs['token_type_ids']
        #inputs = {'input_ids': input_ids, 'attention_mask': attention_mask, 'token_type_ids': token_type_ids, 'labels': labels}
        output = self.model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, labels=labels) 
        return output """
       

""" class Parallel_Classifier(nn.Module):
    def __init__(self, model_name, tokenizer, model, output, dropout):
        super(Parallel_Classifier, self).__init__()
        self.model_name = model_name
        self.tokenizer = tokenizer
        self.model1 = copy.deepcopy(model)
        self.model2 = copy.deepcopy(model)
        self.output = output
        self.mlp = MLP(model.config.hidden_size*2, model.config.hidden_size*2, 3, 3, dropout)

    def forward(self, sentence1, sentence2):
        input1 = self.tokenizer(sentence1, return_tensors='pt', padding=True, truncation=True)
        input2 = self.tokenizer(sentence2, return_tensors='pt', padding=True, truncation=True)
        input1 = input1.to('cuda')
        input2 = input2.to('cuda')
        output1 = self.model1(**input1)
        output2 = self.model2(**input2)
        embeddings1 = output1.last_hidden_state[:,0,:]
        embeddings2 = output2.last_hidden_state[:,0,:]
        x = torch.cat([embeddings1, embeddings2], dim=1)
        x = self.mlp(x)
        return x """