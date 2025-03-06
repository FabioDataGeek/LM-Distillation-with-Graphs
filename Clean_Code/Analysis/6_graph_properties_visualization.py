import pickle as pkl
import pandas as pd
from utils import *
import warnings

# Suppress specific FutureWarning
warnings.filterwarnings("ignore", category=FutureWarning)

DATASET_LABELS = {
    'ag-news': 4,
    'sst-2': 2,
}

dataset = 'ag-news'

columns = [
    'Graph ID', 'Graph_type', 'Classification', 'Classification Outcome', 'num_nodes', 'num_edges', 'avg_degree', 'max_degree', 'min_degree',
    'degree_variance', 'degree_skewness', 'num_weakly_connected_components',
    'num_strongly_connected_components', 'diameter', 'radius', 'avg_path_length',
    'avg_clustering', 'transitivity', 'avg_betweenness', 'max_betweenness',
    'avg_closeness', 'max_closeness', 'avg_eigenvector', 'max_eigenvector',
    'num_communities_louvain', 'modularity_louvain', 'algebraic_connectivity',
    'spectral_radius', 'spectral_gap', 'cycle_count', 'degree_assortativity',
    'girth', 'is_planar', '3_size_pattern_count', '4_size_pattern_count',
    '5_size_pattern_count', 'k3_count_maximal', 'k4_count_maximal', 'k5_count_maximal'
]

df = pd.DataFrame(columns=columns)

# Main loop for processing data
for i in tqdm(range(DATASET_LABELS[dataset]), desc='Processing data', colour='green'):
    for modality in tqdm(['graph', 'subgraph', 'difference_graph'], desc='Processing modalities', colour='blue'):
        for correctness in ['correct', 'incorrect']:
            with open(f'/usrvol/experiments/properties/{dataset}/{i}/{modality}/{correctness}/properties.pkl', 'rb') as f:
                data = pkl.load(f)

            for key, value in data.items():
                row = get_row(key, value, i, correctness, modality)
                # Use pd.concat instead of append
                df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)

# Debug: Check object columns and their unique values
print("Object columns and their unique values:")
for col in df.select_dtypes(include='object').columns:
    unique_values = df[col].dropna().unique()
    print(f"{col}: {unique_values}")

# Detect and fix bool-like object columns
bool_like_columns = [
    col for col in df.select_dtypes(include='object').columns
    if set(df[col].dropna().unique()).issubset({True, False})
]
print(f"Boolean-like columns detected: {bool_like_columns}")
df[bool_like_columns] = df[bool_like_columns].astype(bool)

# Convert numeric columns to appropriate dtype
columns_to_convert = [
    'num_nodes', 'num_edges', 'max_degree', 'min_degree', 'num_weakly_connected_components',
    'num_strongly_connected_components', 'num_communities_louvain', '3_size_pattern_count',
    '4_size_pattern_count', '5_size_pattern_count', 'k3_count_maximal', 'k4_count_maximal',
    'k5_count_maximal', 'diameter', 'radius', 'avg_path_length', 'transitivity',
    'modularity_louvain', 'cycle_count', 'girth'
]
df[columns_to_convert] = df[columns_to_convert].apply(pd.to_numeric, errors='coerce')

# Generate summaries and subdataframes
summaries = generate_summaries(dataset, df, DATASET_LABELS)
subdataframes = generate_subdataframes(dataset, df, DATASET_LABELS)

# Save summaries and plot data
save_summaries(summaries, dataset, DATASET_LABELS)
heatmap(subdataframes, (20, 10), dataset, DATASET_LABELS)
histograms(subdataframes, dataset, DATASET_LABELS)
