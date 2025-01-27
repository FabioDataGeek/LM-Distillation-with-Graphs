import pickle as pkl
import numpy as np
from tqdm import tqdm
from utils import *
from arguments import *
import os

dataset = "sst2"
with open(f"/home/coder/autogoal/explainability_results/summary_{dataset}.pkl", "rb") as f:
    Summary = pkl.load(f)
    
important_words = {}
unimportant_words = {}
important_triples = {}
unimportant_triples = {}
label_list = []


for key, value in tqdm(Summary.items()):
    number = key
    try:
        prediction = value['prediction']
        label = value['label']
    except:
        continue

    # Create necessary directories
    if not os.path.exists(f"/home/coder/autogoal/labels/{dataset}/{prediction}/correct"):
        os.makedirs(f"/home/coder/autogoal/labels/{dataset}/{prediction}/correct")

    if not os.path.exists(f"/home/coder/autogoal/labels/{dataset}/{prediction}/incorrect"):
        os.makedirs(f"/home/coder/autogoal/labels/{dataset}/{prediction}/incorrect")

    # Initialize dictionaries for each prediction class
    if prediction not in label_list:
        label_list.append(prediction)

    if important_words.get(prediction) is None:
        important_words[prediction] = {'correct': [], 'incorrect': []}

    if unimportant_words.get(prediction) is None:
        unimportant_words[prediction] = {'correct': [], 'incorrect': []}

    if important_triples.get(prediction) is None:
        important_triples[prediction] = {'correct': [], 'incorrect': []}

    if unimportant_triples.get(prediction) is None:
        unimportant_triples[prediction] = {'correct': [], 'incorrect': []}

    # Add data based on correctness of the prediction
    if prediction == label:
        important_words[prediction]['correct'].extend(Summary[number]['important_words']) if 'important_words' in Summary[number] else None
        unimportant_words[prediction]['correct'].extend(Summary[number]['unimportant_words']) if 'unimportant_words' in Summary[number] else None

        # Add triples for correct predictions
        important_graph = Summary[number]['important_graph']
        unimportant_graph = Summary[number]['unimportant_graph']

        for edges in important_graph.edges():
            label_0 = important_graph.nodes[edges[0]].get('label')
            label_1 = important_graph.nodes[edges[1]].get('label')

            if label_0 and label_1:
                important_triples[prediction]['correct'].append((label_0, 'Constituency Relation', label_1))

        for edges in unimportant_graph.edges():
            label_0 = unimportant_graph.nodes[edges[0]].get('label')
            label_1 = unimportant_graph.nodes[edges[1]].get('label')

            if label_0 and label_1:
                unimportant_triples[prediction]['correct'].append((label_0, 'Constituency Relation', label_1))
    else:
        important_words[prediction]['incorrect'].extend(Summary[number]['important_words']) if 'important_words' in Summary[number] else None
        unimportant_words[prediction]['incorrect'].extend(Summary[number]['unimportant_words']) if 'unimportant_words' in Summary[number] else None

        # Add triples for incorrect predictions
        important_graph = Summary[number]['important_graph']
        unimportant_graph = Summary[number]['unimportant_graph']

        for edges in important_graph.edges():
            label_0 = important_graph.nodes[edges[0]].get('label')
            label_1 = important_graph.nodes[edges[1]].get('label')

            if label_0 and label_1:
                important_triples[prediction]['incorrect'].append((label_0, 'Constituency Relation', label_1))

        for edges in unimportant_graph.edges():
            label_0 = unimportant_graph.nodes[edges[0]].get('label')
            label_1 = unimportant_graph.nodes[edges[1]].get('label')

            if label_0 and label_1:
                unimportant_triples[prediction]['incorrect'].append((label_0, 'Constituency Relation', label_1))

            
for label in label_list:
    with open(f"/home/coder/autogoal/labels/{dataset}/{label}/correct/important_words.pkl", "wb") as f:
        pkl.dump(important_words[label]['correct'], f)
        
    with open(f"/home/coder/autogoal/labels/{dataset}/{label}/incorrect/important_words.pkl", "wb") as f:    
        pkl.dump(important_words[label]['incorrect'], f)
    
    with open(f"/home/coder/autogoal/labels/{dataset}/{label}/correct/unimportant_words.pkl", "wb") as f:
        pkl.dump(unimportant_words[label]['correct'], f)
    
    with open(f"/home/coder/autogoal/labels/{dataset}/{label}/incorrect/unimportant_words.pkl", "wb") as f:
        pkl.dump(unimportant_words[label]['incorrect'], f)
    
    with open(f"/home/coder/autogoal/labels/{dataset}/{label}/correct/important_triples.pkl", "wb") as f:
        pkl.dump(important_triples[label]['correct'], f)
    
    with open(f"/home/coder/autogoal/labels/{dataset}/{label}/incorrect/important_triples.pkl", "wb") as f:
        pkl.dump(important_triples[label]['incorrect'], f)
    
    with open(f"/home/coder/autogoal/labels/{dataset}/{label}/correct/unimportant_triples.pkl", "wb") as f:
        pkl.dump(unimportant_triples[label]['correct'], f)
    
    with open(f"/home/coder/autogoal/labels/{dataset}/{label}/incorrect/unimportant_triples.pkl", "wb") as f:
        pkl.dump(unimportant_triples[label]['incorrect'], f)
        