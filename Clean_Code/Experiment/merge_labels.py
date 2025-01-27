import os
from tqdm import tqdm
import pickle as pkl
import json
from collections import defaultdict


# Generate the data index for a specific file
def generate_index(data):
    index = defaultdict(dict)
    for item in data:
        key = (item['epoch'], item['dataset'], item['data_index'])
        index[key] = item
    return index


# Get the specific label for a given data index
def get_label(index, epoch, subfolder, data_index):
    key = (epoch, subfolder, data_index)
    matching_element = index.get(key)
    if matching_element is not None:
        return matching_element.get('predicted_label')  # Safely access 'predicted_label'
    else:
        return None


# Merge the labels from the predictions file with the processed tensors
def merge_labels(dataset, epoch):
    # Load predictions JSON
    predictions_path = f"/usrvol/results/{dataset}/predictions.json"
    if not os.path.exists(predictions_path):
        print(f"Predictions file not found: {predictions_path}")
        return

    with open(predictions_path) as f:
        predictions = json.load(f)

    # Generate an index for efficient lookup
    index = generate_index(predictions)

    # Process each graph file in the dataset
    base_folder = f"processed_tensors/{dataset}"
    if not os.path.exists(base_folder):
        print(f"Folder not found: {base_folder}")
        return

    subfolders = [sf for sf in os.listdir(base_folder) if os.path.isdir(os.path.join(base_folder, sf))]

    for subfolder in subfolders:
        subfolder_path = os.path.join(base_folder, subfolder)

        # Get the list of files from the subfolder (graph files are individual files here)
        graph_files = sorted(
            [gf for gf in os.listdir(subfolder_path) if gf.endswith('.pkl')],
            key=lambda x: int(x.split('.')[0])
        )

        if not graph_files:
            continue  # Skip if there are no .pkl files

        # Determine the length of each graph file (assuming the first file gives length info)
        with open(os.path.join(subfolder_path, graph_files[0]), 'rb') as f:
            example = pkl.load(f)
        files_length = len(example)

        # Iterate through all graph files
        for graph_file in tqdm(graph_files, desc=f"Processing {subfolder_path}"):
            graph_file_path = os.path.join(subfolder_path, graph_file)

            # Extract file number for indexing
            file_number = int(graph_file.split('.')[0])

            # Load the graph list from the .pkl file
            with open(graph_file_path, 'rb') as f:
                graph_list = pkl.load(f)

            new_graph_list = []

            # Process each graph in the list
            for i, (graph, labels) in enumerate(graph_list):
                # Calculate real index for the item
                if file_number != 0:
                    real_index = i + file_number * files_length
                else:
                    real_index = i

                # Get the label from the index using subfolder instead of subdataset
                lm_label = get_label(index, epoch, subfolder, real_index)

                # Append the new tuple, including the lm_label (even if it is None)
                new_graph_list.append((graph, labels, lm_label))

            # Create results folder if it doesn't exist
            results_folder = f"{subfolder_path}_with_lm_labels"
            if not os.path.exists(results_folder):
                os.makedirs(results_folder)

            # Save the new graph list with added lm_labels
            results_file_path = os.path.join(results_folder, graph_file)
            with open(results_file_path, 'wb') as f:
                pkl.dump(new_graph_list, f)


# Example usage
if __name__ == "__main__":
    datasets = [
        {"name": "SetFit/ag_news", "epoch": 4},
        {"name": "stanfordnlp/sst2", "epoch": 2}
    ]

    for dataset in datasets:
        merge_labels(dataset["name"], dataset["epoch"])
