import pickle as pkl
import numpy as np
from tqdm import tqdm
from experiment.Analysis.utils import *
from arguments import *

dataset = "sst2"
LABELS_DICT = {
    "ag-news": 4,
    "sst2": 2,
}


number_of_labels = LABELS_DICT[dataset]

for number in tqdm(range(number_of_labels)):
    
    depth_counter_correct = {}
    with open(f"/home/coder/autogoal/depth/{dataset}/{number}/correct/depth_{dataset}.pkl", 'rb') as f:
        correct_depth = pkl.load(f)
    
    correct_depth = {k: v for k, v in sorted(correct_depth.items(), key=lambda item: item[1], reverse=True)}
    
    for depth in correct_depth.values():
        if depth in depth_counter_correct:
            depth_counter_correct[depth] += 1
        else:
            depth_counter_correct[depth] = 1
    
    save_depth_count_plot(depth_counter_correct, 10, f"/home/coder/autogoal/depth/{dataset}/{number}/correct/depth_count.html", f"/home/coder/autogoal/depth/{dataset}/{number}/correct/depth_count.pdf", "Top Correct Depths")
    
    depth_counter_incorrect = {}
    with open(f"/home/coder/autogoal/depth/{dataset}/{number}/incorrect/depth_{dataset}.pkl", 'rb') as f:
        incorrect_depth = pkl.load(f)
    
    incorrect_depth = {k: v for k, v in sorted(incorrect_depth.items(), key=lambda item: item[1], reverse=True)}
    
    for depth in incorrect_depth.values():
        if depth in depth_counter_incorrect:
            depth_counter_incorrect[depth] += 1
        else:
            depth_counter_incorrect[depth] = 1
            
       
    save_depth_count_plot(depth_counter_incorrect, 10, f"/home/coder/autogoal/depth/{dataset}/{number}/incorrect/depth_count.html", f"/home/coder/autogoal/depth/{dataset}/{number}/incorrect/depth_count.pdf", "Top Incorrect Depths")