from transformers import AutoTokenizer
import torch
from torch_geometric.utils import add_remaining_self_loops
import pickle as pkl
from dicts import *

REMOVE_LIST = ['.', '..', '...']

def load_special_tokens(tokenizer):
    """
    Loads the list of special tokens to ignore when getting embeddings.
    """
    return [tokenizer.convert_tokens_to_ids(getattr(tokenizer, token)) for token in tokenizer.SPECIAL_TOKENS_ATTRIBUTES if tokenizer.convert_tokens_to_ids(getattr(tokenizer, token)) is not None]

def token_mapper(tokenizer, tokenized_sentences):
    """
    Maps each word in tokenized sentences to their respective tokens.

    Args:
        tokenizer: The tokenizer used for encoding the sentences.
        tokenized_sentences: List of tokenized sentences.

    Returns:
        A list of dictionaries, where each dictionary maps word indices to token indices.
    """
    # List of tokens to ignore when getting the embeddings
    ignore_token_ids = []
    for special_token_attr in tokenizer.SPECIAL_TOKENS_ATTRIBUTES:
        special_token_id = tokenizer.convert_tokens_to_ids(getattr(tokenizer, special_token_attr))
        if special_token_id is not None:
            ignore_token_ids.append(special_token_id)

    token_mappings = []

    for tokenized_sentence in tokenized_sentences:
        word_to_token_map = {}
        word_index = 0
        token_index = 0
        first_word = True

        while token_index < len(tokenized_sentence):
            token_id = tokenized_sentence[token_index]

            # Skip special tokens
            if token_id in ignore_token_ids:
                token_index += 1
                continue

            token_text = tokenizer.decode([token_id])

            # Handle initial tokens of words (non-subword tokens)
            if not token_text.startswith('##'):
                # Skip dot tokens
                if token_text == '.':
                    token_index += 1
                    continue
                
                # Update word index for subsequent words
                if not first_word:
                    word_index += 1

                # Create a new mapping for the current word
                word_to_token_map[word_index] = [token_index]
                first_word = False
            else:
                # Handle subword tokens (those starting with '##')
                word_to_token_map[word_index].append(token_index)

            token_index += 1

        token_mappings.append(word_to_token_map)

    return token_mappings

def get_words_from_graph(graph):
    """
    Extract words from graph nodes, skipping nodes with special characters.

    Args:
        graphs: List of graph objects.

    Returns:
        List of sentences constructed from graph nodes.
    """
    graph_sentences = []
    # Collect all nodes that are not enclosed by special characters
    sentence_list = [node for node in graph.label[0] if not (node.startswith('«') and node.endswith('»'))]
    graph_sentences.append(' '.join(sentence_list))
    return graph_sentences

def get_tensors_for_sentence(sentence, model, tokenizer, cuda=True):
    """
    Generate tensor representations for each word in the given sentence.

    Args:
        sentence: Input sentence for which embeddings are to be generated.
        model: Pre-trained language model.
        tokenizer: Tokenizer corresponding to the model.
        reduce_method: Method to reduce subword embeddings ('mean', 'sum', 'max').
        cuda: Boolean flag to indicate if CUDA should be used.

    Returns:
        List of tensors for each word in the input sentence.
    """
    # Tokenize the input sentence and convert it to tensors
    inputs = tokenizer(sentence, return_tensors="pt", padding='max_length', max_length=150)
    if cuda:
        inputs = inputs.to('cuda')
    
    # Get the token to word mappings
    mappings = token_mapper(tokenizer, inputs['input_ids'])
    
    # Get model outputs without computing gradients
    with torch.no_grad():
        outputs = model(inputs['input_ids'], attention_mask=inputs['attention_mask'], token_type_ids=inputs['token_type_ids'])
        last_hidden_states = outputs[0]
    
    # Prepare the list to store word tensors
    batch_tensors = []
    
    for i, mapping in enumerate(mappings):
        word_tensors = []
        for word_idx, token_indices in mapping.items():
            # Collect all token tensors for a word
            token_tensors = [last_hidden_states[i, token_idx, :].squeeze(0) for token_idx in token_indices]
            
            # Reduce the token tensors if there are multiple tokens for a word
            if len(token_tensors) > 1:
                token_stack = torch.stack(token_tensors)
                reduced_tensor = torch.mean(token_stack, dim=0)
            else:
                reduced_tensor = token_tensors[0]
            
            word_tensors.append(reduced_tensor)
        batch_tensors.append(word_tensors)
    
    return batch_tensors

def count_tokens(sentence, tokenizer):
    """
    Count the number of tokens in the sentence.

    Args:
        sentence: The input sentence.
        tokenizer: The tokenizer to be used.

    Returns:
        Number of tokens in the sentence.
    """
    inputs = tokenizer(sentence, return_tensors="pt")
    return inputs['input_ids'].shape[1]

def remove_punctuation_nodes(graph, ids_to_remove):
    """
    Removes the nodes corresponding to punctuation marks from the graph.

    Args:
        graph: The graph from which nodes are to be removed.
        ids_to_remove: List of node IDs to be removed.
    """
    ids_to_remove = torch.tensor(ids_to_remove)
    # Create a mask to filter out edges connected to the punctuation nodes
    not_in_ids = ~torch.isin(graph.edge_index[0], ids_to_remove)
    mask = torch.logical_and(not_in_ids, ~torch.isin(graph.edge_index[1], ids_to_remove))
    # Update edge indices and labels based on the mask
    graph.edge_index = graph.edge_index[:, mask]
    graph.edge_label = [val for val, keep in zip(graph.edge_label[0], mask.tolist()) if keep]

def process_edge_labels(graph, mode):
    """
    Process edge labels of the graph based on the selected mode.

    Args:
        graph: The input graph whose edge labels need processing.
        mode: The mode of processing (e.g., 'sintactic', 'semantic').

    Returns:
        The fill value for adding self-loops.
    """
    mode_dicts = {
        'sintactic': (SINTACTIC_DICT, SINTACTIC_NUM_DICT),
        'semantic': (SEMANTIC_DICT, SEMANTIC_NUM_DICT),
        'constituency': (CONSTITUENCY_DICT, CONSTITUENCY_NUM_DICT),
        'sintactic+semantic': (SIN_SEM_DICT, SIN_SEM_NUM_DICT),
        'sintactic+constituency': (SIN_CON_DICT, SIN_CON_NUM_DICT),
        'semantic+constituency': (SEM_CON_DICT, SEM_CON_NUM_DICT),
        'sintactic+semantic+constituency': (SIN_SEM_CON_DICT, SIN_SEM_CON_NUM_DICT)
    }
    if mode not in mode_dicts:
        raise NotImplementedError

    if len(graph.edge_label) != 0:
        dict_map, num_dict = mode_dicts[mode]
        # Update edge labels based on the mapping dictionary
        graph.edge_label = torch.tensor([num_dict[dict_map[str(label)]] for label in graph.edge_label])
        return len(num_dict)
    else:
        return len(mode_dicts[mode][1])

def torch_geometric_tensors(sentence_tensors, specific_tensors_route, graphs, mode):
    """
    Update graph features and edge labels for torch geometric processing.

    Args:
        sentence_tensors: List of tensors corresponding to the sentences.
        graphs: List of graph objects.
        mode: The mode to determine how edge labels should be processed.

    Returns:
        Updated graph object for torch geometric processing.
    """

    # Load pre-saved tensors specific to the model
    with open(specific_tensors_route, 'rb') as f:
        MODEL_SPECIFIC_TENSORS = pkl.load(f)

    feature_list, node_list, ids_to_remove = [], [], []
    

    # Iterate over each graph and update features and nodes
    for i in range(len(sentence_tensors)):
        tensor_list, dict_nodes = [], {}
    
        for j, label in enumerate(graphs[i].label):
            # Handle special tokens
            if label.startswith('«') and label.endswith('»'):
                tensor_list.append(MODEL_SPECIFIC_TENSORS[label].to('cuda'))
            elif sentence_tensors[i]:
                # Append tensor from sentence tensors if available
                tensor_list.append(sentence_tensors[i].pop(0))
            else:
                # Use zero tensor if no sentence tensors are available
                tensor_list.append(torch.zeros(768).to('cuda'))
            dict_nodes[j] = str(label)

            # Mark punctuation nodes for removal
            if label in REMOVE_LIST:
                ids_to_remove.append(j)

        feature_list.append(torch.stack(tensor_list))
        node_list.append(dict_nodes)

    # Concatenate features from all graphs
    features = torch.cat(feature_list, dim=0)
    graphs.x = features
    graphs.dict_nodes = node_list
    graphs.num_nodes = features.size(0)

    # Remove punctuation nodes from the graph
    if len(graphs.edge_label) != 0:
        remove_punctuation_nodes(graphs, ids_to_remove)
    # Process edge labels based on the specified mode
    fill_value = process_edge_labels(graphs, mode)

    # Add remaining self-loops to the graph
    graphs.edge_index, graphs.edge_attr = add_remaining_self_loops(
        graphs.edge_index, num_nodes=graphs.x.size(0), edge_attr=graphs.edge_label, fill_value=fill_value
    )
    # Update the batch attribute of the graph
    graphs.batch = torch.full((graphs.x.size(0),), graphs.batch[0].item(), dtype=torch.long)
    return graphs
