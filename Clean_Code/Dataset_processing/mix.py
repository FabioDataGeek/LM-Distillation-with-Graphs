from mixer import *
import pickle as pkl
import os
import json
from tqdm import tqdm

folder = os.listdir("/usrvol/processed_data")
mixer = GraphMixer()

for dataset in tqdm(folder, colour='white'):
    print(f"Processing {dataset}...")
    subfolder = os.listdir(f"/usrvol/processed_data/{dataset}")

    for sub in tqdm(subfolder, colour='blue'):
        print(f"Processing {sub}...")
        sin_folder = os.listdir(f"/usrvol/processed_data/{dataset}/{sub}/sintactic")
        sem_folder = os.listdir(f"/usrvol/processed_data/{dataset}/{sub}/semantic")
        cons_folder = os.listdir(f"/usrvol/processed_data/{dataset}/{sub}/constituency")
        # get sintactic+semantic
        print("Getting sintactic+semantic...")
        
        for graphs in tqdm(sin_folder, colour='green'):
            if graphs.startswith("sintactic"):
                if graphs.endswith(".pkl"):
                    sintactic = pkl.load(open(f"/usrvol/processed_data/{dataset}/{sub}/sintactic/{graphs}", "rb"))
                    number = graphs.split("sintactic")[-1].split(".")[0]
                    semantic = pkl.load(open(f"/usrvol/processed_data/{dataset}/{sub}/semantic/semantic{number}.pkl", "rb"))
            
            assert len(sintactic) == len(semantic), f"The number of graphs must be the same: {len(sintactic)} != {len(semantic)}."
            new_list = []
            for graphs in range(len(sintactic)):
                graph_list1 = [sintactic[graphs][0], semantic[graphs][0]]
                graph_list2 = [sintactic[graphs][1], semantic[graphs][1]]
                labels = sintactic[graphs][2]
                assert sintactic[graphs][2] == semantic[graphs][2], f"The labels must be the same: {sintactic[graphs][2]} != {semantic[graphs][2]}."

                G1 = mixer.mix(graph_list1, sintactic=True, semantic=True)
                G2 = mixer.mix(graph_list2, sintactic=True, semantic=True)
                new_list.append((G1, G2, labels))

            if not os.path.exists(f"/usrvol/processed_data/{dataset}/{sub}/sintactic+semantic"):
                os.makedirs(f"/usrvol/processed_data/{dataset}/{sub}/sintactic+semantic")

            with open(f"/usrvol/processed_data/{dataset}/{sub}/sintactic+semantic/sintactic_semantic{number}.pkl", 'wb') as f:
                pkl.dump(new_list, f)

        # get sintactic+constituency
        print("Getting sintactic+constituency...")
              
        for graphs in tqdm(sin_folder, colour='red'):
            if graphs.startswith("sintactic"):
                if graphs.endswith(".pkl"):
                    sintactic = pkl.load(open(f"/usrvol/processed_data/{dataset}/{sub}/sintactic/{graphs}", "rb"))
                    number = graphs.split("sintactic")[-1].split(".")[0]
                    constituency = pkl.load(open(f"/usrvol/processed_data/{dataset}/{sub}/constituency/constituency{number}.pkl", "rb"))
            
            assert len(sintactic) == len(constituency), "The number of graphs must be the same."
            new_list = []
            for graphs in range(len(sintactic)):
                graph_list1 = [sintactic[graphs][0], constituency[graphs][0]]
                graph_list2 = [sintactic[graphs][1], constituency[graphs][1]]
                labels = sintactic[graphs][2]
                assert sintactic[graphs][2] == constituency[graphs][2], "The labels must be the same."

                G1 = mixer.mix(graph_list1, sintactic=True, constituency=True)
                G2 = mixer.mix(graph_list2, sintactic=True, constituency=True)
                new_list.append((G1, G2, labels))

            if not os.path.exists(f"/usrvol/processed_data/{dataset}/{sub}/sintactic+constituency"):
                os.makedirs(f"/usrvol/processed_data/{dataset}/{sub}/sintactic+constituency")

            with open(f"/usrvol/processed_data/{dataset}/{sub}/sintactic+constituency/sintactic_constituency{number}.pkl", 'wb') as f:
                pkl.dump(new_list, f)

        # get semantic+constituency
        print("Getting semantic+constituency...")
        for graphs in tqdm(sem_folder, colour='yellow'):
            if graphs.startswith("semantic"):
                if graphs.endswith(".pkl"):
                    semantic = pkl.load(open(f"/usrvol/processed_data/{dataset}/{sub}/semantic/{graphs}", "rb"))
                    number = graphs.split("semantic")[-1].split(".")[0]
                    constituency = pkl.load(open(f"/usrvol/processed_data/{dataset}/{sub}/constituency/constituency{number}.pkl", "rb"))
            
            assert len(semantic) == len(constituency), "The number of graphs must be the same."
            new_list = []
            for graphs in range(len(semantic)):
                graph_list1 = [semantic[graphs][0], constituency[graphs][0]]
                graph_list2 = [semantic[graphs][1], constituency[graphs][1]]
                labels = semantic[graphs][2]
                assert semantic[graphs][2] == constituency[graphs][2], "The labels must be the same."

                G1 = mixer.mix(graph_list1, semantic=True, constituency=True)
                G2 = mixer.mix(graph_list2, semantic=True, constituency=True)
                new_list.append((G1, G2, labels))

            if not os.path.exists(f"/usrvol/processed_data/{dataset}/{sub}/semantic+constituency"):
                os.makedirs(f"/usrvol/processed_data/{dataset}/{sub}/semantic+constituency")

            with open(f"/usrvol/processed_data/{dataset}/{sub}/semantic+constituency/semantic_constituency{number}.pkl", 'wb') as f:
                pkl.dump(new_list, f)


        # get sintactic+semantic+constituency
        print("Getting sintactic+semantic+constituency...")
        for graphs in tqdm(sin_folder, colour='cyan'):
            if graphs.startswith("sintactic"):
                if graphs.endswith(".pkl"):
                    sintactic = pkl.load(open(f"/usrvol/processed_data/{dataset}/{sub}/sintactic/{graphs}", "rb"))
                    number = graphs.split("sintactic")[-1].split(".")[0]
                    semantic = pkl.load(open(f"/usrvol/processed_data/{dataset}/{sub}/semantic/semantic{number}.pkl", "rb"))
                    constituency = pkl.load(open(f"/usrvol/processed_data/{dataset}/{sub}/constituency/constituency{number}.pkl", "rb"))
            
            assert len(sintactic) == len(semantic) == len(constituency), "The number of graphs must be the same."
            new_list = []
            for graphs in range(len(sintactic)):
                graph_list1 = [sintactic[graphs][0], semantic[graphs][0], constituency[graphs][0]]
                graph_list2 = [sintactic[graphs][1], semantic[graphs][1], constituency[graphs][1]]
                labels = sintactic[graphs][2]
                assert sintactic[graphs][2] == semantic[graphs][2] == constituency[graphs][2], "The labels must be the same."

                G1 = mixer.mix(graph_list1, sintactic=True, semantic=True, constituency=True)
                G2 = mixer.mix(graph_list2, sintactic=True, semantic=True, constituency=True)
                new_list.append((G1, G2, labels))

            if not os.path.exists(f"/usrvol/processed_data/{dataset}/{sub}/sintactic+semantic+constituency"):
                os.makedirs(f"/usrvol/processed_data/{dataset}/{sub}/sintactic+semantic+constituency")

            with open(f"/usrvol/processed_data/{dataset}/{sub}/sintactic+semantic+constituency/sintactic_semantic_constituency{number}.pkl", 'wb') as f:
                pkl.dump(new_list, f)
