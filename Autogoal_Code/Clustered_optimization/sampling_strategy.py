import numpy as np
import random
from collections import defaultdict
from scipy.spatial.distance import pdist, squareform
from sklearn.cluster import KMeans
from experiment.Optimization.architecture_GNNs import *
from experiment.Clustered_optimization.dataloader import *
from experiment.Optimization.arguments import *


# Assuming graph_sizes is a dictionary with graph IDs as keys and graph sizes as values
# and cluster_ids is a numpy array with cluster assignments.

# Stratified sampling by size

def load_dataset():
    dataset = Dataset_GNN(
        root=args["root_test_data_path"], files_path=args["raw_test_data_path"]
    )
    return dataset

from collections import defaultdict
import random

def stratified_sampling(clusters, cluster_id, num_samples=20):
    """
    Perform stratified sampling on graphs with a specific cluster ID.

    Args:
        clusters (list): List of cluster IDs for each graph.
        cluster_id (int): Target cluster ID to sample from.
        num_samples (int): Total number of graphs to sample.

    Returns:
        list: List of sampled graph objects.
        list: List of sampled graph IDs.
    """
    # Step 1: Group graphs by size bins
    size_bins = defaultdict(list)
    clustered_graphs = {}

    # Iterate through the dataset and group by size
    dataset = load_dataset()  # Load the specific dataset
    for i in range(len(dataset)):
        for k, graph in enumerate(dataset[i]):
            graph_id = i * 256 + k
            if graph_id < len(clusters):  # Ensure graph ID is within range
                if clusters[graph_id] == cluster_id:  # Match cluster ID
                    size = graph.x.size(0)
                    clustered_graphs[graph_id] = size
                    size_bin = size // 5  # Group by size ranges of 5
                    size_bins[size_bin].append(graph_id)
            else:
                continue
    # Step 2: Proportional sampling from size bins
    sampled_graphs_ids = []
    total_graphs = sum(len(bin_graphs) for bin_graphs in size_bins.values())
    for bin_graphs in size_bins.values():
        bin_sample_size = max(1, round(num_samples * len(bin_graphs) / total_graphs))
        sampled_graphs_ids.extend(random.sample(bin_graphs, min(bin_sample_size, len(bin_graphs))))

    # Ensure final sample does not exceed num_samples
    if len(sampled_graphs_ids) > num_samples:
        sampled_graphs_ids = random.sample(sampled_graphs_ids, num_samples)

    # Sort sampled graph IDs for reproducibility
    sampled_graphs_ids.sort()

    # Step 3: Retrieve the graph objects
    final_graph_list = []
    for graph_id in sampled_graphs_ids:
        dataset_id = graph_id // 256
        specific_graph_id = graph_id % 256
        dataset = load_dataset()
        graphs = dataset[dataset_id]
        graph = graphs[specific_graph_id]  # Directly access the specific graph
        final_graph_list.append(graph)

    return final_graph_list, list(sampled_graphs_ids)


def weighted_random_sampling(clusters, cluster_id, num_samples=20):
    """
    Perform weighted random sampling on graphs with a specific cluster ID.

    Args:
        dataset (list): A list of datasets.
        clusters (list): List of cluster IDs for each graph.
        cluster_id (int): Target cluster ID to sample from.
        num_samples (int): Total number of graphs to sample.

    Returns:
        list: List of sampled graph objects.
        list: List of sampled graph IDs.
    """
    clustered_graphs = {}

    # Step 1: Collect graphs and sizes for the target cluster
    dataset = load_dataset()
    for i in range(len(dataset)):
        for k, graph in enumerate(dataset[i]):
            graph_id = i * 256 + k
            if graph_id < len(clusters):
                if clusters[graph_id] == cluster_id:
                    size = graph.x.size(0)
                    clustered_graphs[graph_id] = size
                else:
                    continue

    # Step 2: Perform weighted random sampling
    graph_ids = list(clustered_graphs.keys())
    sizes = np.array([clustered_graphs[graph_id] for graph_id in graph_ids])
    probabilities = sizes / sizes.sum()  # Compute probabilities
    sampled_graph_ids = np.random.choice(graph_ids, size=min(num_samples, len(graph_ids)), replace=False, p=probabilities)

    # Step 3: Retrieve the graph objects
    final_graph_list = []
    for graph_id in sampled_graph_ids:
        dataset_id = graph_id // 256
        specific_graph_id = graph_id % 256
        dataset = load_dataset()
        graphs = dataset[dataset_id]
        graph = graphs[specific_graph_id]
        final_graph_list.append(graph)

    return final_graph_list, list(sampled_graph_ids)


def maximal_diversity_sampling(clusters, cluster_id, num_samples=20):
    """
    Perform maximal diversity sampling on graphs with a specific cluster ID.

    Args:
        dataset (list): A list of datasets.
        clusters (list): List of cluster IDs for each graph.
        cluster_id (int): Target cluster ID to sample from.
        num_samples (int): Total number of graphs to sample.

    Returns:
        list: List of sampled graph objects.
        list: List of sampled graph IDs.
    """
    clustered_graphs = {}

    # Step 1: Collect graphs and sizes for the target cluster
    dataset = load_dataset()
    for i in range(len(dataset)):
        for k, graph in enumerate(dataset[i]):
            graph_id = i * 256 + k
            if graph_id < len(clusters):
                if clusters[graph_id] == cluster_id:
                    size = graph.x.size(0)
                    clustered_graphs[graph_id] = size
            else:
                continue
    # Step 2: Compute pairwise distances and select diverse graphs
    graph_ids = list(clustered_graphs.keys())
    sizes = np.array([clustered_graphs[graph_id] for graph_id in graph_ids]).reshape(-1, 1)
    distances = squareform(pdist(sizes, metric='euclidean'))  # Compute pairwise distances
    
    # Greedy selection for diversity
    sampled_graph_ids = []
    sampled_indices = set()
    for _ in range(min(num_samples, len(graph_ids))):
        if not sampled_indices:
            # Start with a random graph
            idx = np.random.choice(len(graph_ids))
        else:
            # Select the graph farthest from the current selection
            idx = np.argmax(distances[list(sampled_indices)].sum(axis=0))
        sampled_indices.add(idx)
        sampled_graph_ids.append(graph_ids[idx])

    # Step 3: Retrieve the graph objects
    final_graph_list = []
    for graph_id in sampled_graph_ids:
        dataset_id = graph_id // 256
        specific_graph_id = graph_id % 256
        dataset = load_dataset()
        graphs = dataset[dataset_id]
        graph = graphs[specific_graph_id]
        final_graph_list.append(graph)

    return final_graph_list, list(sampled_graph_ids)


def secondary_clustering_sampling(clusters, cluster_id, num_samples=20, n_subclusters=5):
    """
    Perform secondary clustering sampling on graphs with a specific cluster ID.

    Args:
        dataset (list): A list of datasets.
        clusters (list): List of cluster IDs for each graph.
        cluster_id (int): Target cluster ID to sample from.
        num_samples (int): Total number of graphs to sample.
        n_subclusters (int): Number of sub-clusters to create.

    Returns:
        list: List of sampled graph objects.
        list: List of sampled graph IDs.
    """
    clustered_graphs = {}

    # Step 1: Collect graphs and sizes for the target cluster
    dataset = load_dataset()
    for i in range(len(dataset)):
        for k, graph in enumerate(dataset[i]):
            graph_id = i * 256 + k
            if graph_id < len(clusters):
                if clusters[graph_id] == cluster_id:
                    size = graph.x.size(0)
                    clustered_graphs[graph_id] = size
            else:
                continue

    # Step 2: Perform secondary clustering
    graph_ids = list(clustered_graphs.keys())
    sizes = np.array([clustered_graphs[graph_id] for graph_id in graph_ids]).reshape(-1, 1)
    n_subclusters = min(len(graph_ids), n_subclusters)  # Adjust sub-clusters if necessary
    kmeans = KMeans(n_clusters=n_subclusters, random_state=42)
    subcluster_ids = kmeans.fit_predict(sizes)
    
    # Step 3: Sample evenly across sub-clusters
    subcluster_to_graphs = {i: [] for i in range(n_subclusters)}
    for graph_id, subcluster_id in zip(graph_ids, subcluster_ids):
        subcluster_to_graphs[subcluster_id].append(graph_id)

    sampled_graph_ids = []
    for graphs_in_subcluster in subcluster_to_graphs.values():
        count = min(num_samples - len(sampled_graph_ids), len(graphs_in_subcluster))
        sampled_graph_ids.extend(random.sample(graphs_in_subcluster, count))

    # Step 4: Retrieve the graph objects
    final_graph_list = []
    for graph_id in sampled_graph_ids:
        dataset_id = graph_id // 256
        specific_graph_id = graph_id % 256
        dataset = load_dataset()
        graphs = dataset[dataset_id]
        graph = graphs[specific_graph_id]
        final_graph_list.append(graph)

    return final_graph_list, list(sampled_graph_ids)
