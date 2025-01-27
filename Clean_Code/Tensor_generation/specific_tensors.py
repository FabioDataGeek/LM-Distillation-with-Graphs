from transformers import AutoTokenizer, BertModel, AutoModelForMaskedLM, AutoModelForPreTraining, AutoModel, ElectraTokenizer
import torch
import pickle as pkl    


""" Here, we are going to create the specific tensors that indicates sintactic, semantic
and constituency relations and nodes. We will create these tensors for each of the encoders
tested """

# Make this work for BERT, DeBERTa, AlBERT and ELECTRA

# Here we get the tensor from [CLS] token
def get_specific_tensor(term, model, tokenizer):
    inputs = tokenizer(term, return_tensors="pt")
    
    with torch.no_grad():
        outputs = model(**inputs)  # inputs['input_ids'], attention_mask=inputs['attention_mask'], token_type_ids=inputs['token_type_ids']
        last_hidden_states = outputs[0]
        final_tensor = last_hidden_states[:,0,:].squeeze(0)
    return final_tensor

with open('/usrvol/utils/term_list.pkl', 'rb') as f:
    terms = pkl.load(f)

for model_name in ['bert-base-uncased', 'albert/albert-base-v2', 'microsoft/deberta-base', 'google/electra-base-discriminator']:
    try:
        output = model_name.split('/')[1]
    except:
        output = model_name
    tensor_dict = {}
    
    if model_name in ['bert-base-uncased', 'albert/albert-base-v2']:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name)
    elif model_name in ['microsoft/deberta-base']:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name)
    elif model_name in ['google/electra-base-discriminator']:
        tokenizer = ElectraTokenizer.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name)
    
    model.eval()
    for term in terms:
        tensor_dict[term] = get_specific_tensor(term, model, tokenizer)
    with open(f'/usrvol/processed_data/{output}_specific_tensors.pkl', 'wb') as f:
        pkl.dump(tensor_dict, f)

    torch.cuda.empty_cache()
