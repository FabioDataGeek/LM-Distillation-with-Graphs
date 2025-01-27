import pickle as pkl
import numpy as np
from tqdm import tqdm
from experiment.Analysis.utils import *
from arguments import *
import os

dataset = "sst2"

with open(f"/home/coder/autogoal/summary_{dataset}.pkl", "rb") as f:
    Summary = pkl.load(f)
    
depth_dict = {}
label_list = []
for key, value in tqdm(Summary.items()):
    number = key
    try:
        label = value['label']
        prediction = value['prediction']
        depth = value['depth']
    except:
        continue
    
    if not label in label_list:
        label_list.append(label)
    
    if not os.path.exists(f"/home/coder/autogoal/depth/{dataset}/{prediction}/correct"):
            os.makedirs(f"/home/coder/autogoal/depth/{dataset}/{prediction}/correct")
        
    if not os.path.exists(f"/home/coder/autogoal/depth/{dataset}/{prediction}/incorrect"):
        os.makedirs(f"/home/coder/autogoal/depth/{dataset}/{prediction}/incorrect")
    
    if label == prediction:
        depth_dict[prediction] = {'correct': {}, 'incorrect': {}} if depth_dict.get(prediction) is None else depth_dict[prediction]
        depth_dict[prediction]['correct'][number] = depth
    
    else:
        depth_dict[prediction] = {'correct': {}, 'incorrect': {}} if depth_dict.get(prediction) is None else depth_dict[prediction]
        depth_dict[prediction]['incorrect'][number] = depth
        

for label in label_list:
    with open(f"/home/coder/autogoal/depth/{dataset}/{label}/correct/depth_{dataset}_{label}.pkl", "wb") as f:
        pkl.dump(depth_dict[label]['correct'], f)
        
    with open(f"/home/coder/autogoal/depth/{dataset}/{label}/incorrect/depth_{dataset}_{label}.pkl", "wb") as f:
        pkl.dump(depth_dict[label]['incorrect'], f)
        
    
        