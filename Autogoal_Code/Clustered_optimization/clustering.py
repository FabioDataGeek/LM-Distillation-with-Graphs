from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
from experiment.Optimization.arguments import *
from experiment.Optimization.architecture_GNNs import *
from experiment.Clustered_optimization.dataloader import *
import numpy as np
from tqdm import tqdm

def load_dataset():
    dataset = Dataset_GNN(
        root=args["root_test_data_path"], files_path=args["raw_test_data_path"]
    )
    return dataset

dataset = load_dataset()

graph_sizes = []
for i in tqdm(range(len(dataset)), desc='Processing dataset...'):
    graph_list = dataset[i]
    for k in range(len(graph_list)):
        graph_sizes.append(graph_list[k].x.size(0))
# Assume X is your dataset of graphs represented by node counts in different groups

graph_sizes = np.array(graph_sizes).reshape(-1, 1)

wcss = []
silhouette_scores = []
K = range(2, 10)  # Range of k to try

for k in tqdm(K, desc='Generating clusters...'):
    kmeans = KMeans(n_clusters=k, random_state=42).fit(graph_sizes)
    wcss.append(kmeans.inertia_)
    silhouette_scores.append(silhouette_score(graph_sizes, kmeans.labels_))

plt.ion()

# Plot Elbow Method
plt.figure(figsize=(10, 6), dpi=600)
plt.plot(K, wcss, marker='o', linestyle='-', color='b', linewidth=2, markersize=8)
plt.xticks(ticks=K, fontsize=12)
plt.yticks(fontsize=12)
plt.xlabel('Number of clusters (k)', fontsize=16, weight='bold')
plt.ylabel('Within-Cluster Sum of Squares', fontsize=16, weight='bold')
plt.title('Elbow Method For Optimal k', fontsize=24, weight='bold')
plt.grid(True, linestyle='--', linewidth=0.5)
plt.tight_layout()
plt.savefig('elbow_method.png', format='png', dpi=600)
plt.close()

# Plot Silhouette Scores
plt.figure(figsize=(10, 6), dpi=600)
plt.plot(K, silhouette_scores, marker='o', linestyle='-', color='b', linewidth=2, markersize=8)
plt.xticks(ticks=K, fontsize=12)
plt.yticks(fontsize=12)
plt.xlabel('Number of clusters (k)', fontsize=16, weight='bold')
plt.ylabel('Average Silhouette Score', fontsize=16, weight='bold')
plt.title('Silhouette Analysis For Optimal k', fontsize=24, weight='bold')
plt.grid(True, linestyle='--', linewidth=0.5)
plt.tight_layout()
plt.savefig('silhouette_scores.png', format='png', dpi=600)
plt.close()

print('done!')
