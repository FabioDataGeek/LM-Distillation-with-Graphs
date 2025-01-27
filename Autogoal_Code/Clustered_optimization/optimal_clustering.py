import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from experiment.Optimization.arguments import *
from experiment.Optimization.architecture_GNNs import *
from experiment.Clustered_optimization.dataloader import *
from tqdm import tqdm

def load_dataset():
    dataset = Dataset_GNN(
        root=args["root_test_data_path"], files_path=args["raw_test_data_path"]
    )
    return dataset

def label_clusters(number_of_clusters, dataset_length):

    dataset = load_dataset()
    graph_sizes = {}
    initial_length = dataset_length
    for i in tqdm(range(len(dataset)), desc='Processing dataset...'):
        graph_list = dataset[i]
        for k in range(len(graph_list)):
            graph_sizes[i*initial_length + k] = graph_list[k].x.size(0)

    graphs_features = list(graph_sizes.values())

    # Assuming graph_sizes is already defined and contains the graph sizes
    graphs_features = np.array(graphs_features).reshape(-1, 1)

    # Use the optimal number of clusters (e.g., k=3)
    optimal_k = number_of_clusters
    kmeans = KMeans(n_clusters=optimal_k, random_state=42)
    cluster_ids = kmeans.fit_predict(graphs_features)  # Predict cluster IDs
    # Save the dataset with cluster IDs
    np.save(f"cluster_ids.npy", cluster_ids)

    # Visualize the clusters
    plt.figure(figsize=(11, 6.6), dpi=600)  # Reduced y-axis size by changing figsize
    for cluster in range(optimal_k):
        # Mask to select points belonging to the current cluster
        mask = cluster_ids == cluster
        cluster_points = graphs_features[mask]  # Select points based on the mask
        plt.scatter(cluster_points, np.zeros_like(cluster_points), label=f'Cluster {cluster}', s=50, alpha=0.7)

    # Plot centroids for better visualization
    centroids = kmeans.cluster_centers_
    plt.scatter(centroids, [0] * len(centroids), s=200, c='red', marker='x', label='Centroids')
    plt.xlabel('Graph Size', fontsize=16, fontweight='bold')
    plt.title('Obtained Clusters', fontsize=24, fontweight='bold')
    plt.xticks(fontsize=12)
    plt.yticks([])  # Remove y-axis labels for clarity
    plt.legend(fontsize=12)
    plt.grid(True, linestyle='--', linewidth=0.5)
    plt.tight_layout()
    plt.savefig('clusters.png', format='png', dpi=600)
    plt.close()