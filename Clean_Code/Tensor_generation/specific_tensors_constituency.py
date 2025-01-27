from transformers import AutoTokenizer, BertModel, AutoModelForMaskedLM, AutoModelForPreTraining, AutoModel, ElectraTokenizer
from transformers import AutoModelForSequenceClassification, AutoTokenizer, AutoConfig, get_linear_schedule_with_warmup
import torch
import pickle as pkl    
from tqdm import tqdm
from arguments import *
""" Here, we are going to create the specific tensors that indicates sintactic, semantic
and constituency relations and nodes. We will create these tensors for each of the encoders
tested """

LABELS = {
    'ag_news': 4,
    'sst2': 2
}

args = arguments()

# Here we get the tensor from [CLS] token
def get_specific_tensor(term, model, tokenizer):
    inputs = tokenizer(term, return_tensors="pt")
    
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)  # inputs['input_ids'], attention_mask=inputs['attention_mask'], token_type_ids=inputs['token_type_ids']
        hidden_states = outputs.hidden_states
        last_hidden_state = hidden_states[-1]
        final_tensor = last_hidden_state[:,0,:].squeeze(0)
    return final_tensor

with open('/usrvol/utils/term_list.pkl', 'rb') as f:
    terms = pkl.load(f)

for dataset in tqdm(['ag_news', 'sst2']):
    tensor_dict = {}
    model_route = f"/usrvol/results/{dataset}/best_fine_tuned_model.pt"
    model = AutoModelForSequenceClassification.from_pretrained('google-bert/bert-base-uncased', num_labels=LABELS[dataset])
    state_dict = torch.load(model_route)
    model.load_state_dict(state_dict)
    tokenizer = AutoTokenizer.from_pretrained('google-bert/bert-base-uncased') 
    model.eval()
    for term in terms:
        tensor_dict[term] = get_specific_tensor(term, model, tokenizer)
    with open(f'/usrvol/processed_data/{dataset}_specific_tensors.pkl', 'wb') as f:
        pkl.dump(tensor_dict, f)

    torch.cuda.empty_cache()
