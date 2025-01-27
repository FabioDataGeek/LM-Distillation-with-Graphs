from transformers import AutoTokenizer
import torch
from torch_geometric.utils.convert import from_networkx
import pickle as pkl
from torch_geometric.utils import add_remaining_self_loops, add_self_loops
from dicts import *


with open('/usrvol/utils/albert-base-v2_specific_tensors.pkl', 'rb') as f:
    albert_base_v2_specific_tensors = pkl.load(f)

with open('/usrvol/utils/bert-base-uncased_specific_tensors.pkl', 'rb') as f:
    bert_base_uncased_specific_tensors = pkl.load(f)

with open('/usrvol/utils/deberta-base_specific_tensors.pkl', 'rb') as f:
    deberta_base_specific_tensors = pkl.load(f)

with open('/usrvol/utils/electra-base-discriminator_specific_tensors.pkl', 'rb') as f:
    electra_base_specific_tensors = pkl.load(f)


REMOVE_LIST = ['.', '..', '...']

# This function will give us a map of every word with their tokens in the language model
def token_mapper(tokenizer, tokenized_sentences):
    # List of tokens to ignore when getting the embeddings
    ignore_tokens = []
    for token in tokenizer.SPECIAL_TOKENS_ATTRIBUTES:
        if tokenizer.convert_tokens_to_ids(getattr(tokenizer, token)) is None:
            continue
        else:
            ignore_tokens.append(tokenizer.convert_tokens_to_ids(getattr(tokenizer, token)))

    token_mapping_list = []
    for tokenized_sentence in tokenized_sentences:
        token_mapping = {}
        word_iterator = 0
        token_iterator = 0
        first_iteration = True
        for i, token in enumerate(tokenized_sentence):
            if token in ignore_tokens:
                token_iterator += 1
            else:
                token = tokenizer.decode(token)
                # This means it is the initial token of a word
                if not '##' in token:
                    token_id = tokenizer.convert_tokens_to_ids(token)
                    dot_token = tokenizer.convert_tokens_to_ids('.')
                    # We check if the token is not a dot
                    if token_id == dot_token:
                        token_iterator += 1
                        continue
                    # on the first iteration we don't update the word index, we start from 0
                    if not first_iteration:
                        word_iterator += 1
                    token_mapping[word_iterator] = [token_iterator]
                    token_iterator += 1
                
                # when dealing with ## tokens
                else:
                    token_mapping[word_iterator].append(token_iterator)
                    token_iterator += 1
                first_iteration = False
        token_mapping_list.append(token_mapping)
    return token_mapping_list


# Use this function to get the sentence from the graph
def get_words_from_graph(graphs):
    batch_sentence = []
    for graph in range(len(graphs)):
        sentence_list = []
        for node in graphs[graph].label:
            if node.startswith('«') and node.endswith('»'):
                continue
            else:
                sentence_list.append(node)
        sentence = ' '.join(sentence_list)
        batch_sentence.append(sentence)
    return batch_sentence

# here we give the sentence from the previous function and the model and tokenizer
def get_tensors_for_sentence(sentence, model, tokenizer, reduce_method='mean', cuda=True):
    
    inputs = tokenizer(sentence, return_tensors="pt", padding='max_length', max_length=150)
    if cuda:
        inputs = inputs.to('cuda')
    mappings = token_mapper(tokenizer, inputs['input_ids'])
    with torch.no_grad():
        outputs = model(inputs['input_ids'], attention_mask=inputs['attention_mask'], token_type_ids=inputs['token_type_ids'])
        last_hidden_states = outputs[0]
    batch_tensors = []
    for i in range(len(mappings)):
        mapping = mappings[i]
        final_tensors = []
        for word in mapping:
            tensor_list = []
            for token in mapping[word]:
                tensor_list.append(last_hidden_states[i,token,:].squeeze(0))
            if len(tensor_list) > 1:
                tensor_list = torch.stack(tensor_list)
                if reduce_method == 'mean':
                    reduced_tensor = torch.mean(tensor_list, dim=0)
                elif reduce_method == 'sum':
                    reduced_tensor = torch.sum(tensor_list, dim=0)
                elif reduce_method == 'max':
                    reduced_tensor = torch.max(tensor_list, dim=0).values
                else:
                    raise NotImplementedError
            else:
                reduced_tensor = tensor_list[0]

            final_tensors.append(reduced_tensor)
        batch_tensors.append(final_tensors)
    return batch_tensors

def count_tokens(sentence, tokenizer):
    inputs = tokenizer(sentence, return_tensors="pt")
    number_of_tokens = inputs['input_ids'].shape[1]
    return number_of_tokens

def torch_geometric_tensors(sentence_tensors, graphs, model_name, mode):
    if model_name == 'bert-base-uncased':
        TENSORS = bert_base_uncased_specific_tensors
    elif model_name == 'albert/albert-base-v2':
        TENSORS = albert_base_v2_specific_tensors
    elif model_name == 'microsoft/deberta-base':
        TENSORS = deberta_base_specific_tensors
    elif model_name == 'google/electra-base-discriminator':
        TENSORS = electra_base_specific_tensors

    feature_list = []
    node_list = []
    ids_to_remove = []
    # Iterate over each graph
    for i in range(len(sentence_tensors)):
        tensor_list = []
        dict_nodes = {}
        tensors = graphs[i]
        for j, label in enumerate(tensors.label):
            if str(label).startswith('«') and str(label).endswith('»'):
                tensor_list.append(TENSORS[label].to('cuda'))
                dict_nodes[j] = str(label)
            else:
                if len(sentence_tensors[i]) != 0:
                    tensor_list.append(sentence_tensors[i][0])
                    sentence_tensors[i] = sentence_tensors[i][1:]
                    dict_nodes[j] = str(label)
                else:
                    tensor_list.append(torch.zeros(768).to('cuda'))
                    dict_nodes[j] = str(label)

        for j, label in enumerate(tensors.label):
            if label in REMOVE_LIST:
                ids_to_remove.append(j)
        try:
            feature_list.append(torch.stack(tensor_list))
        except:
            feature_list.append(torch.stack([torch.zeros(768).to('cuda')]))
        node_list.append(dict_nodes)
    features = torch.cat(feature_list, dim=0)
    graphs.x = features


    graphs.dict_nodes = node_list
    graphs.num_nodes = features.size(0)


    # This is for removing the edges associated with the punctuation nodes
    ids_to_remove = torch.tensor(ids_to_remove)
    mask = torch.logical_and(~torch.isin(graphs.edge_index[0], ids_to_remove), ~torch.isin(graphs.edge_index[1], ids_to_remove))   
    graphs.edge_index = graphs.edge_index[:, mask]
    mask = mask.tolist()
    if len(graphs.edge_label) != 0:
        graphs.edge_label[0] = [val for val, keep in zip(graphs.edge_label[0], mask) if keep]
    

        if mode == 'sintactic':
            graphs.edge_label = torch.tensor([SINTACTIC_NUM_DICT[SINTACTIC_DICT[str(graphs.edge_label[0][i])]] for i in range(len(graphs.edge_label[0]))])
            fill_value = len(SINTACTIC_NUM_DICT)
        elif mode == 'semantic':
            graphs.edge_label = torch.tensor([SEMANTIC_NUM_DICT[SEMANTIC_DICT[str(graphs.edge_label[0][i])]] for i in range(len(graphs.edge_label[0]))])
            fill_value = len(SEMANTIC_NUM_DICT)
        elif mode == 'constituency':
            graphs.edge_label = torch.tensor([CONSTITUENCY_NUM_DICT['Constituency relation']]*graphs.edge_index.shape[1])
            fill_value = len(CONSTITUENCY_NUM_DICT)
        elif mode == 'sintactic+semantic':
            graphs.edge_label = torch.tensor([SIN_SEM_NUM_DICT[SIN_SEM_DICT[str(graphs.edge_label[0][i])]] for i in range(len(graphs.edge_label[0]))])
            fill_value = len(SIN_SEM_NUM_DICT)
        elif mode == 'sintactic+constituency':
            graphs.edge_label = torch.tensor([SIN_CON_NUM_DICT[SIN_CON_DICT[str(graphs.edge_label[0][i])]] for i in range(len(graphs.edge_label[0]))])
            fill_value = len(SIN_CON_NUM_DICT)
        elif mode == 'semantic+constituency':
            graphs.edge_label = torch.tensor([SEM_CON_NUM_DICT[SEM_CON_DICT[str(graphs.edge_label[0][i])]] for i in range(len(graphs.edge_label[0]))])
            fill_value = len(SEM_CON_NUM_DICT)
        elif mode == 'sintactic+semantic+constituency':
            graphs.edge_label = torch.tensor([SIN_SEM_CON_NUM_DICT[SIN_SEM_CON_DICT[str(graphs.edge_label[0][i])]] for i in range(len(graphs.edge_label[0]))])
            fill_value = len(SIN_SEM_CON_NUM_DICT)
        else:
            raise NotImplementedError
    
    else:
        if mode == 'sintactic':
            graphs.edge_label = torch.tensor([])
            fill_value = len(SINTACTIC_NUM_DICT)
        elif mode == 'semantic':
            graphs.edge_label = torch.tensor([])
            fill_value = len(SEMANTIC_NUM_DICT)
        elif mode == 'constituency':
            graphs.edge_label = torch.tensor([])
            fill_value = len(CONSTITUENCY_NUM_DICT)
        elif mode == 'sintactic+semantic':
            graphs.edge_label = torch.tensor([])
            fill_value = len(SIN_SEM_NUM_DICT)
        elif mode == 'sintactic+constituency':
            graphs.edge_label = torch.tensor([])
            fill_value = len(SIN_CON_NUM_DICT)
        elif mode == 'semantic+constituency':
            graphs.edge_label = torch.tensor([])
            fill_value = len(SEM_CON_NUM_DICT)
        elif mode == 'sintactic+semantic+constituency':
            graphs.edge_label = torch.tensor([])
            fill_value = len(SIN_SEM_CON_NUM_DICT)
        else:
            raise NotImplementedError

    graphs.edge_index, graphs.edge_attr = add_remaining_self_loops(graphs.edge_index, num_nodes=graphs.x.size(0), edge_attr=graphs.edge_label, fill_value=fill_value)
    graphs.batch = torch.tensor([graphs.batch[0].item()]*graphs.x.size(0))
    return graphs


