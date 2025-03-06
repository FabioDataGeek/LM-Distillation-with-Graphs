import networkx as nx
from collections import deque
import plotly.graph_objects as go
from torch_geometric.utils import to_networkx
import copy
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
import plotly.express as px
import plotly.io as pio
pio.kaleido.scope.default_format = "pdf"
import matplotlib.pyplot as plt
from networkx.algorithms import isomorphism
import seaborn as sns
import os
from tqdm import tqdm


LABEL_MAPPER = {
    "ag-news": {
        0: "World",
        1: "Sports",
        2: "Business",
        3: "Sci-Tech"
    },
    "sst-2": {
        0: "Negative",
        1: "Positive"
    }
}

def safe_degree_assortativity_coefficient(G):
    """
    Safely compute the degree assortativity coefficient for a graph.
    Returns NaN if the graph is not suitable for computation.
    """
    if G.number_of_edges() == 0:
        # No edges, assortativity is undefined
        return float('nan')
    try:
        return nx.degree_assortativity_coefficient(G)
    except (ZeroDivisionError, RuntimeWarning, nx.NetworkXError):
        # Catch specific errors and return NaN
        return float('nan')

def get_labeled_subgraph(pyg_graph, coalition):
    
    # Convert the graph to a networkx graph, and clean it
    graph = to_networkx(pyg_graph)
    graph.remove_edges_from(nx.selfloop_edges(graph))
    graph = remove_cross_links(graph)
    graph = remove_isolated_nodes(graph)
    
    # Get the subgraph 
    subgraph, _ = get_subgraph(graph, coalition)
    subgraph = remove_isolated_nodes(subgraph)
    
    # Add labels to the nodes
    terms = pyg_graph.dict_nodes[0]
    for node in subgraph.nodes():
        subgraph.nodes[node]['label'] = terms[node]
    
    return subgraph


def get_subgraph(G, coalition):
    # Step 1: Identify important subgraph
    G_copy = G.copy()
    bfs_distances = bfs_from_node(G_copy, source=0)

    # Find the minimum distance from the source to any coalition node
    try:
        min_distance = min(bfs_distances[node] for node in coalition if node in bfs_distances)
    except ValueError:
        min_distance = 0
        
    # Remove nodes not in coalition with a distance <= min_distance
    nodes_to_remove = [node for node, dist in bfs_distances.items() if node not in coalition and dist <= min_distance]
    G_copy.remove_nodes_from(nodes_to_remove)

    # Keep only components containing coalition nodes
    components = list(nx.weakly_connected_components(G_copy))
    nodes_to_keep = set()
    for component in components:
        if any(node in coalition for node in component):
            nodes_to_keep.update(component)

    subgraph = G_copy.subgraph(nodes_to_keep).copy()
    
    return subgraph, min_distance

def remove_cross_links(G, root=0):
    """
    1. Create a BFS tree T from the given root in graph G.
    2. Identify cross-links (edges in G but not in T).
    3. Remove certain cross-links based on custom logic:
       - Remove cross-links from all but the smallest-ID predecessor, etc.
    4. Remove isolated nodes from the resulting graph.
    5. Return the cleaned-up DiGraph.
    """
    # --- Make a copy so we don't alter the original graph ---
    G_copy = G.copy()
    
    # --- Create a BFS tree from the root node ---
    T = nx.bfs_tree(G_copy, source=root)

    # --- Identify cross-links (edges in G_copy but not in T) ---
    cross_links = [edge for edge in G_copy.edges() if edge not in T.edges()]

    links_to_remove = []
    for edge in cross_links:
        src, target = edge
        predecessors = list(G_copy.predecessors(target))
        if predecessors:
            predecessors_copy = predecessors.copy()
            
            # Remove all predecessors with ID greater than target
            for predecessor in predecessors:
                if predecessor > target:
                    links_to_remove.append((predecessor, target))
                    predecessors_copy.remove(predecessor)
            
            # Sort remaining predecessors in ascending order
            predecessors_copy.sort(reverse=False)
            
            # Remove all but the last predecessor
            for predecessor in predecessors_copy[:-1]:
                links_to_remove.append((predecessor, target))

    # --- Remove duplicates and then remove them from G_copy ---
    links_to_remove = list(set(links_to_remove))
    G_copy.remove_edges_from(links_to_remove)

    # --- Remove isolated nodes (if any) ---
    G_copy = remove_isolated_nodes(G_copy)

    return G_copy

def remove_isolated_nodes(graph):
    isolates = list(nx.isolates(graph))
    graph.remove_nodes_from(isolates)
    return graph

def find_subgraph_isomorphisms(G, pattern):
    """
    Find *all* subgraph isomorphisms of 'pattern' inside 'G'.
    Returns a list of frozensets, each frozenset being the node set in G
    that matches the pattern.
    """
    GM = isomorphism.GraphMatcher(G, pattern)
    matches = []
    for sub_mapping in GM.subgraph_isomorphisms_iter():
        match_nodes = frozenset(sub_mapping.values())
        matches.append(match_nodes)
    # Remove duplicates
    return list(set(matches))


def count_subgraph(G, pattern):
    """Return the number of distinct subgraphs in G isomorphic to pattern."""
    return len(find_subgraph_isomorphisms(G, pattern))


def count_cliques_of_size_k(G, k):
    """
    Counts how many cliques of size k exist in G.
    Uses the built-in find_cliques, which enumerates maximal cliques,
    so this will only catch size-k *maximal* cliques. If you want *all*
    size-k cliques (including those contained in bigger cliques), you might
    need a different approach or to post-process the cliques.
    """
    count = 0
    for clique in nx.find_cliques(G):
        if len(clique) == k:
            count += 1
    return count


def bfs_from_node(G, source):
    """
    Perform BFS from specific node and return the shortest path lengths.
    
    Parameters:
    - G (nx.Graph or nx.DiGraph): The graph.
    - source (int): The source node for BFS.
    
    Returns:
    - dict: Dictionary mapping node to its shortest path length from source.
    """
    visited = {}
    queue = deque([source])
    visited[source] = 0

    while queue:
        current = queue.popleft()
        current_dist = visited[current]
        for neighbor in G.neighbors(current):
            if neighbor not in visited:
                visited[neighbor] = current_dist + 1
                queue.append(neighbor)
    
    return visited


def find_all_predecessors(node, graph):
    predecessors = list(graph.predecessors(node))
    if len(predecessors) == 0:
        return []
    else:
        for parent in predecessors:
            return predecessors + find_all_predecessors(parent, graph)

def semantic_features(G, coalition, features):
    """
    Retrieves the sequence of words corresponding to the important subgraph,
    focusing only on leaf nodes (nodes without successors), sorted by their numeric IDs.

    Parameters:
    - G (nx.DiGraph): The original directed graph.
    - coalition (list of int): List of coalition node indices.
    - features (dict): Node features for embeddings.

    Returns:
    - important_words (list of str): Words from the important subgraph (leaves only).
    - unimportant_words (list of str): Words from the unimportant subgraph.
    - important_sequence (list of str): Ordered sequence of words from the important subgraph.
    - word_embeddings (dict): Dictionary mapping words to embeddings.
    - special_embeddings (dict): Dictionary mapping special labels to embeddings.
    - G_important (nx.DiGraph): Subgraph of relevant nodes.
    - G_unimportant (nx.DiGraph): Subgraph of irrelevant nodes.
    - Depth (int): Minimum distance from the source node to any coalition node.
    """


    word_embeddings = {}
    special_embeddings = {}

    G_important, depth = get_subgraph(G, coalition)

    # Step 2: Extract leaf nodes and sort by numeric value
    leaf_nodes = [node for node in G_important.nodes if G_important.out_degree(node) == 0]
    leaf_nodes.sort()  # Sort by numeric value (node IDs are ordered as per word order in the sentence)

    # Retrieve words from leaf nodes
    important_sequence = []
    important_words = []

    for leaf in leaf_nodes:
        label = G_important.nodes[leaf].get('label', "")
        if "«" not in label and "»" not in label:
            important_words.append(label)
            important_sequence.append(label)  # Ordered by numeric node IDs
            word_embeddings[label] = features[leaf].detach().cpu().numpy()
        else:
            special_embeddings[label] = features[leaf].detach().cpu().numpy()

    # Step 3: Identify the unimportant subgraph
    total_nodes = set(G.nodes())
    total_edges = set(G.edges())
    subgraph_nodes = set(G_important.nodes())
    subgraph_edges = set(G_important.edges())

    G_unimportant = nx.DiGraph()
    
    # Add nodes to G_unimportant, preserving their attributes
    for node in total_nodes - subgraph_nodes:
        G_unimportant.add_node(node, **G.nodes[node])

    # Add edges to G_unimportant, preserving their attributes
    G_unimportant.add_edges_from(total_edges - subgraph_edges)

    unimportant_words = []
    for node in G_unimportant.nodes():
        label = G_unimportant.nodes[node].get('label', "")
        if "«" not in label and "»" not in label:
            unimportant_words.append(label)
            word_embeddings[label] = features[node].detach().cpu().numpy()
        else:
            special_embeddings[label] = features[node].detach().cpu().numpy()

    return important_words, unimportant_words, important_sequence, word_embeddings, special_embeddings, G_important, G_unimportant, depth


def count_leaves(G, root):
    """
    Returns a dictionary mapping each node to the number of leaves in its subtree.
    """
    leaves_count = {}
    def dfs(node, parent):
        neighbors = [n for n in G.neighbors(node) if n != parent]
        if not neighbors:
            # Leaf node
            leaves_count[node] = 1
            return 1
        else:
            total_leaves = 0
            for neighbor in neighbors:
                total_leaves += dfs(neighbor, node)
            leaves_count[node] = total_leaves
            return total_leaves
    dfs(root, None)
    return leaves_count


def hierarchy_pos(G, root, leaves_count, width=10.0, vert_gap=1.0, vert_loc=0, xcenter=0.5, pos=None, parent=None):
    """
    Modified hierarchy_pos to allocate more space to nodes with larger subtrees.
    Positions nodes with the root at the top.
    Increased width and vert_gap to significantly increase horizontal and vertical separation between nodes.
    """
    if pos is None:
        pos = {}
    pos[root] = (xcenter, vert_loc)
    neighbors = [n for n in G.neighbors(root) if n != parent]
    if neighbors:
        total_leaves = sum(leaves_count[child] for child in neighbors)
        dx = width / total_leaves
        next_x = xcenter - width / 2 - dx / 2
        for neighbor in neighbors:
            num_leaves = leaves_count[neighbor]
            child_width = dx * num_leaves
            next_x += dx * num_leaves / 2
            pos = hierarchy_pos(G, neighbor, leaves_count, width=child_width, vert_gap=vert_gap,
                                vert_loc=vert_loc - vert_gap, xcenter=next_x, pos=pos, parent=root)
            next_x += dx * num_leaves / 2
    return pos


def visualize_graph(data, graph_idx, result_idx, root):
    # Get coalition and original graph
    coalition = data[graph_idx][0][result_idx]['coalition']
    original_graph = data[graph_idx][2]
    
    # Access node labels from original_graph.dict_nodes()
    node_labels_dict = original_graph.dict_nodes[0]  # Assuming it returns a dict {node_number: label_str}
    
    # Convert the PyTorch Geometric graph to a NetworkX graph
    G = to_networkx(original_graph, to_undirected=False, remove_self_loops=True)
    
    # Ensure G is a DiGraph
    if isinstance(G, nx.MultiDiGraph):
        ##print("Converting MultiDiGraph to DiGraph.")
        G = nx.DiGraph(G)
    
    G_copy = eliminate_cross_links_and_isolated_nodes(G)
    
    # **Additional Debugging: Verify the structure of G_copy**
    #print("=== Debugging Information ===")
    #print(f"Number of nodes in G_copy: {G_copy.number_of_nodes()}")
    #print(f"Number of edges in G_copy: {G_copy.number_of_edges()}")
    #print(f"Graph type: {type(G_copy)}")
    
    # Check for multiple root nodes
    root_nodes = [node for node, deg in G_copy.in_degree() if deg == 0]
    #print(f"Root Nodes (in-degree 0): {root_nodes}")
    if len(root_nodes) != 1:
        raise ValueError(f"Expected exactly one root node, found {len(root_nodes)}: {root_nodes}")
    
    # Check for nodes with multiple parents
    nodes_with_multiple_parents = [node for node, deg in G_copy.in_degree() if deg > 1]
    if nodes_with_multiple_parents:
        raise ValueError(f"Nodes with multiple parents: {nodes_with_multiple_parents}")
    
    # Check for cycles
    if not nx.is_directed_acyclic_graph(G_copy):
        raise ValueError("The graph contains cycles.")
    
    # Check connectivity
    T_full = nx.bfs_tree(G_copy, source=root_nodes[0])
    if len(T_full.nodes()) != len(G_copy.nodes()):
        disconnected_nodes = set(G_copy.nodes()) - set(T_full.nodes())
        raise ValueError(f"The graph is disconnected. Disconnected nodes: {disconnected_nodes}")
    
    # Final check with is_arborescence
    if not nx.is_arborescence(G_copy):
        #print("=== Edges after cross-link removal ===")
        #print(list(G_copy.edges()))
        raise ValueError("The resulting graph is not an arborescence. Check the cross-link removal logic.")
    
    # Compute the number of leaves under each node
    leaves_count = count_leaves(G_copy, root=root_nodes[0])
    
    # Compute the depth of the tree
    depth = nx.dag_longest_path_length(G_copy)
    
    # Adjust vert_gap to increase vertical separation between nodes
    vert_gap = 2.0 / (depth + 1)  # Increased vertical gap for more spacing between nodes
    
    # Set the vertical location of the root node to the maximum value
    vert_loc = 0  # Root at the top
    
    # Generate positions using the modified hierarchy_pos function
    pos = hierarchy_pos(G_copy, root=root_nodes[0], leaves_count=leaves_count, width=15.0, vert_gap=vert_gap, vert_loc=vert_loc)
    
    # Extract positions and labels for Plotly
    node_x = []
    node_y = []
    node_text = []
    node_color = []
    for node in G_copy.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        
        # Get the label from node_labels_dict, fallback to node number if not found
        node_label = node_labels_dict.get(node, str(node))
        node_text.append(node_label)
        
        # Determine node color based on coalition membership
        if node in coalition:
            node_color.append('red')
        else:
            node_color.append('lightblue')
    
    # Normalize positions to fit within [0, 1] range
    x_min = min(node_x)
    x_max = max(node_x)
    y_min = min(node_y)
    y_max = max(node_y)
    
    x_range = x_max - x_min if x_max != x_min else 1
    y_range = y_max - y_min if y_max != y_min else 1

    # Normalize node positions
    node_x_norm = [(x - x_min) / x_range for x in node_x]
    node_y_norm = [(y - y_min) / y_range for y in node_y]
    
    # Create edge traces
    edge_x = []
    edge_y = []
    for edge in T_full.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        # Normalize positions
        x0 = (x0 - x_min) / x_range
        x1 = (x1 - x_min) / x_range
        y0 = (y0 - y_min) / y_range
        y1 = (y1 - y_min) / y_range

        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])
        
    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=2, color='gray'),
        hoverinfo='none',
        mode='lines')
    
    # Create node traces with labels
    node_trace = go.Scatter(
        x=node_x_norm, y=node_y_norm,
        mode='markers+text',
        text=node_text,
        textposition="top center",
        hoverinfo='text',
        marker=dict(
            color=node_color,
            size=40,  # Increased node size for better visibility
            line=dict(width=3, color='black')
        ),
        textfont=dict(
            size=18,  # Adjusted font size for better readability
            family="Arial, sans-serif",  # Set font family to enhance readability
            color='black'
        )
    )
    
    # Set fixed figure dimensions for interactive display
    fig_width = 3000  # Increased width to better fit larger node sizes and spacing
    fig_height = 1400  # Increased height to reduce blank space
    
    # Create the figure
    fig = go.Figure(data=[edge_trace, node_trace],
                    layout=go.Layout(
                        title=f"Tree Graph with Root Node {root} (Graph Index: {graph_idx}, Result Index: {result_idx})",
                        titlefont_size=20,
                        showlegend=False,
                        hovermode='closest',
                        margin=dict(b=20, l=20, r=20, t=60),  # Increased margins to reduce overlapping
                        width=fig_width,
                        height=fig_height,
                        xaxis=dict(
                            showgrid=False,
                            zeroline=False,
                            showticklabels=False,
                            autorange=True
                        ),
                        yaxis=dict(
                            showgrid=False,
                            zeroline=False,
                            showticklabels=False,
                            autorange='reversed'  # Reverse y-axis to place root at the top
                        )
                    )
                )
    
    # Update modebar to add high-quality download button options
    fig.update_layout(
        modebar_add=['toImage'],
    )
    
    # Define the configuration for high-quality image download
    config = {
        'toImageButtonOptions': {
            'format': 'png',  # Set the format to PNG
            'filename': 'high_quality_tree_plot',  # Set the filename
            'height': 2000,  # Adjusted height to better fit the nodes
            'width': 4500,  # Adjusted width to better fit the nodes
            'scale': 3  # Increase the scale for better quality
        }
    }
    
    # Display the figure interactively with custom configuration
    fig.show(config=config)


def visualize_and_save_embeddings_pca_2d(dataset, clusters_dict, vectors_dict, cluster_labels=None, output_file="pca_visualization_2d.html", image_file=None, pdf_file=None):
    """
    Visualize embeddings in a 2D PCA space and save the visualization as an HTML file, image, or PDF.
    
    Parameters:
        dataset (str): Name of the dataset.
        clusters_dict (dict): Dictionary where each key is a cluster identifier and its value is a list of embedding keys.
        vectors_dict (dict): Dictionary where each key is an identifier and its value is the embedding vector.
        cluster_labels (dict, optional): Dictionary mapping cluster identifiers to descriptive labels.
        output_file (str): Path to save the HTML visualization.
        image_file (str, optional): Path to save the visualization as an image (PNG, JPEG, etc.).
        pdf_file (str, optional): Path to save the visualization as a PDF.
    """
    # Gather all vectors and assign their cluster labels
    embeddings = []
    labels = []
    for cluster, keys in clusters_dict.items():
        for key in keys:
            if key in vectors_dict:
                embeddings.append(vectors_dict[key])
                labels.append(cluster)
    
    # Convert to numpy array
    embeddings = np.array(embeddings)
    
    # Reduce dimensionality with PCA
    pca = PCA(n_components=2)
    reduced_embeddings = pca.fit_transform(embeddings)
    
    # Create a DataFrame for visualization
    data = pd.DataFrame(reduced_embeddings, columns=['PCA1', 'PCA2'])
    
    # Map cluster identifiers to descriptive labels if provided
    if cluster_labels:
        data['Cluster'] = [cluster_labels.get(label, label) for label in labels]
    else:
        data['Cluster'] = labels
    
    # Visualization with Plotly
    fig = px.scatter(
        data, 
        x='PCA1', y='PCA2', 
        color='Cluster',
        title=f"2D PCA Embeddings from {dataset}",
        labels={'Cluster': 'Cluster'},
        opacity=0.8
    )
    
    # Aesthetic improvements
    fig.update_traces(marker=dict(size=6))
    fig.update_layout(
        xaxis_title=None,
        yaxis_title=None,
        legend_title_text='Cluster',
        legend=dict(
            x=0.98,  # Legend position slightly more centered
            y=0.98,  # Legend position slightly lower
            xanchor='right',
            yanchor='top',
            bgcolor='rgba(255, 255, 255, 0.8)',  # Semi-transparent background
            bordercolor='black',
            borderwidth=1
        ),
        title=dict(
            text=f"2D PCA Embeddings from {dataset}",
            x=0.5,  # Center the title horizontally
            xanchor='center',
            font=dict(size=24, family="Arial", color="black", weight="bold")  # Bigger and bold title
        ),
        width=1200,  # Increased width
        height=900   # Increased height
    )
    
    # Save the visualization as an HTML file
    if output_file:
        pio.write_html(fig, file=output_file, auto_open=False)
        print(f"Visualization saved to {output_file}")
    
    # Save as a static image with increased quality
    if image_file:
        pio.write_image(fig, file=image_file, scale=5)  # Further increased scale for higher resolution
        print(f"Static image saved to {image_file}")

    # Save as a PDF file
    if pdf_file:
        pio.write_image(fig, file=pdf_file, format='pdf', scale=10, width=1200, height=900)  # Increased size for PDF
        print(f"PDF saved to {pdf_file}")


def visualize_word_embeddings_2d(important_words, unimportant_words, word_to_embedding, 
                                  output_file="word_embeddings_2d.html", 
                                  image_file=None, pdf_file=None):
    """
    Visualize word embeddings in a 2D PCA space, distinguishing between important and unimportant words.

    Parameters:
        important_words (list): List of important words to visualize.
        unimportant_words (list): List of unimportant words to visualize.
        word_to_embedding (dict): Dictionary mapping words to their embedding vectors.
        output_file (str): Path to save the HTML visualization.
        image_file (str, optional): Path to save the visualization as an image (PNG, JPEG, etc.).
        pdf_file (str, optional): Path to save the visualization as a PDF.
    """
    # Combine words and fetch their embeddings
    words = []
    embeddings = []
    labels = []

    # Add important words and their embeddings
    for word in important_words:
        if word in word_to_embedding:
            words.append(word)
            embeddings.append(word_to_embedding[word])
            labels.append("Important")

    # Add unimportant words and their embeddings
    for word in unimportant_words:
        if word in word_to_embedding:
            words.append(word)
            embeddings.append(word_to_embedding[word])
            labels.append("Unimportant")

    # Check if there are embeddings to process
    if not embeddings:
        raise ValueError("No embeddings found for the provided words.")

    # Convert embeddings to numpy array
    embeddings = np.array(embeddings)

    # Reduce dimensionality with PCA
    pca = PCA(n_components=2)
    try:
        reduced_embeddings = pca.fit_transform(embeddings)
    except:
        return None

    # Create a DataFrame for visualization
    data = pd.DataFrame(reduced_embeddings, columns=['PCA1', 'PCA2'])
    data['Word'] = words
    data['Category'] = labels

    # Visualization with Plotly
    fig = px.scatter(
        data, 
        x='PCA1', y='PCA2', 
        color='Category',
        hover_data=['Word'],
        title="2D PCA Visualization of Word Embeddings",
        labels={'Category': 'Word Category'},
        opacity=0.8
    )

    # Aesthetic improvements
    fig.update_traces(marker=dict(size=8))
    fig.update_layout(
        xaxis_title=None,
        yaxis_title=None,
        legend_title_text='Word Category',
        legend=dict(
            x=0.98, 
            y=0.98, 
            xanchor='right',
            yanchor='top',
            bgcolor='rgba(255, 255, 255, 0.8)',
            bordercolor='black',
            borderwidth=1
        ),
        title=dict(
            text="2D PCA Visualization of Word Embeddings",
            x=0.5, 
            xanchor='center',
            font=dict(size=20, family="Arial", color="black", weight="bold")
        ),
        width=1000, 
        height=800
    )

    # Save the visualization as an HTML file
    if output_file:
        pio.write_html(fig, file=output_file, auto_open=False)
        print(f"Visualization saved to {output_file}")

    # Save as a static image
    if image_file:
        pio.write_image(fig, file=image_file, scale=5)
        print(f"Static image saved to {image_file}")

    # Save as a PDF file
    if pdf_file:
        pio.write_image(fig, file=pdf_file, format='pdf', scale=5, width=1000, height=800)
        print(f"PDF saved to {pdf_file}")
        
        
def visualize_word_embeddings_3d(important_words, unimportant_words, word_to_embedding, 
                                  output_file="word_embeddings_2d.html", 
                                  image_file=None, pdf_file=None):
    """
    Visualize word embeddings in a 3D PCA space, distinguishing between important and unimportant words.

    Parameters:
        important_words (list): List of important words to visualize.
        unimportant_words (list): List of unimportant words to visualize.
        word_to_embedding (dict): Dictionary mapping words to their embedding vectors.
        output_file (str): Path to save the HTML visualization.
        image_file (str, optional): Path to save the visualization as an image (PNG, JPEG, etc.).
        pdf_file (str, optional): Path to save the visualization as a PDF.
    """
    # Combine words and fetch their embeddings
    words = []
    embeddings = []
    labels = []

    # Add important words and their embeddings
    for word in important_words:
        if word in word_to_embedding:
            words.append(word)
            embeddings.append(word_to_embedding[word])
            labels.append("Important")

    # Add unimportant words and their embeddings
    for word in unimportant_words:
        if word in word_to_embedding:
            words.append(word)
            embeddings.append(word_to_embedding[word])
            labels.append("Unimportant")

    # Check if there are embeddings to process
    if not embeddings:
        raise ValueError("No embeddings found for the provided words.")

    # Convert embeddings to numpy array
    embeddings = np.array(embeddings)

    # Reduce dimensionality with PCA
    pca = PCA(n_components=3)
    reduced_embeddings = pca.fit_transform(embeddings)

    # Create a DataFrame for visualization
    data = pd.DataFrame(reduced_embeddings, columns=['PCA1', 'PCA2', 'PCA3'])
    data['Word'] = words
    data['Category'] = labels

    # Visualization with Plotly
    fig = px.scatter_3d(
        data, 
        x='PCA1', y='PCA2', z='PCA3', 
        color='Category',
        hover_data=['Word'],
        title="3D PCA Visualization of Word Embeddings",
        labels={'Category': 'Word Category'},
        opacity=0.8
    )

    # Aesthetic improvements
    fig.update_traces(marker=dict(size=8))
    fig.update_layout(
        xaxis_title=None,
        yaxis_title=None,
        legend_title_text='Word Category',
        legend=dict(
            x=0.98, 
            y=0.98, 
            xanchor='right',
            yanchor='top',
            bgcolor='rgba(255, 255, 255, 0.8)',
            bordercolor='black',
            borderwidth=1
        ),
        title=dict(
            text="3D PCA Visualization of Word Embeddings",
            x=0.5, 
            xanchor='center',
            font=dict(size=20, family="Arial", color="black", weight="bold")
        ),
        width=1000, 
        height=800
    )

    # Save the visualization as an HTML file
    if output_file:
        pio.write_html(fig, file=output_file, auto_open=False)
        print(f"Visualization saved to {output_file}")

    # Save as a static image
    if image_file:
        pio.write_image(fig, file=image_file, scale=5)
        print(f"Static image saved to {image_file}")

    # Save as a PDF file
    if pdf_file:
        pio.write_image(fig, file=pdf_file, format='pdf', scale=5, width=1000, height=800)
        print(f"PDF saved to {pdf_file}")


def save_depth_count_plot(depth_counts, top_n, html_path, pdf_path, title):
    # Sort the depth counts by count in descending order
    sorted_depth = sorted(depth_counts.items(), key=lambda x: x[1], reverse=True)[:top_n]
    labels = [str(depth) for depth, count in sorted_depth]
    counts = [count for _, count in sorted_depth]

    # Create Plotly bar plot
    fig = go.Figure()
    fig.add_trace(go.Bar(x=labels, y=counts, text=counts, textposition='outside'))
    fig.update_layout(
        title=title,
        xaxis_title='Depth',
        yaxis_title='Count',
        xaxis=dict(tickangle=45),
        template="plotly_white",
        margin=dict(l=50, r=50, t=50, b=50),
    )
    
    # Save the plot as an HTML file
    fig.write_html(html_path)

    # Save the plot as a PDF
    fig.write_image(pdf_path, format='pdf')

    
def save_triple_count_plot(triple_counts, top_n, html_path, pdf_path):
    """
    Creates a bar plot for the top N triple counts and saves it as HTML and PDF.
    """
    # Sort the triple counts by count in descending order
    sorted_triples = sorted(triple_counts.items(), key=lambda x: x[1], reverse=True)[:top_n]
    labels = [f"{triple[0]}-{triple[1]}" for triple, count in sorted_triples]
    counts = [count for _, count in sorted_triples]

    # Create Plotly bar plot
    fig = go.Figure()
    fig.add_trace(go.Bar(x=labels, y=counts, text=counts, textposition='outside'))
    fig.update_layout(
        title=f'Top {top_n} Important Triple Counts',
        xaxis_title='Triple (entity1-entity2)',
        yaxis_title='Count',
        xaxis=dict(tickangle=45),
        template="plotly_white",
        margin=dict(l=50, r=50, t=50, b=50),
    )

    # Save the plot as an HTML file
    fig.write_html(html_path)

    # Save the plot as a PDF file
    fig.write_image(pdf_path, format='pdf')


def save_word_count_plot(word_counts, top_n, html_path, pdf_path, title):
    """
    Creates a bar plot for the top N words and saves it as HTML and PDF.
    """
    # Sort the word counts by count in descending order
    sorted_words = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)[:top_n]
    labels = [word for word, count in sorted_words]
    counts = [count for _, count in sorted_words]

    # Create Plotly bar plot
    fig = go.Figure()
    fig.add_trace(go.Bar(x=labels, y=counts, text=counts, textposition='outside'))
    fig.update_layout(
        title=title,
        xaxis_title='Word',
        yaxis_title='Count',
        xaxis=dict(tickangle=45),
        template="plotly_white",
        margin=dict(l=50, r=50, t=50, b=50),
    )

    # Save the plot as an HTML file
    fig.write_html(html_path)

    # Save the plot as a PDF file
    fig.write_image(pdf_path, format='pdf')
    

def graph_degree_features(G):
    """
    Extracts various structural properties (features) from a NetworkX graph.
    
    Parameters
    ----------
    G : nx.Graph
        A NetworkX Graph (assumed undirected in this example).
        
    Returns
    -------
    features : dict
        A dictionary of computed graph properties.
    """

    features = {}

    # ----------------------------------------------------------------------
    # 1. Degree-related measures
    # ----------------------------------------------------------------------
    degrees = [deg for _, deg in G.degree()]
    if len(degrees) == 0:
        # Handle empty graphs gracefully
        features["num_nodes"] = 0
        features["num_edges"] = 0
        features["avg_degree"] = 0
        features["max_degree"] = 0
        features["min_degree"] = 0
        features["degree_variance"] = 0
        features["degree_skewness"] = 0
    else:
        features["num_nodes"] = G.number_of_nodes()
        features["num_edges"] = G.number_of_edges()
        
        # Average degree
        features["avg_degree"] = np.mean(degrees)
        
        # Max / Min degree
        features["max_degree"] = np.max(degrees)
        features["min_degree"] = np.min(degrees)

        # Degree variance
        features["degree_variance"] = np.var(degrees)
        
        # A simple measure of skewness (Pearson’s moment coefficient)
        mean_deg = features["avg_degree"]
        sd_deg = np.std(degrees)
        if sd_deg != 0:
            # skewness = (mean of ((deg - mean_deg)^3)) / sd_deg^3
            skew_num = np.mean([(d - mean_deg)**3 for d in degrees])
            features["degree_skewness"] = skew_num / (sd_deg**3)
        else:
            features["degree_skewness"] = 0.0

    return features


def graph_connectivity_features(G):
    """
    Compute connectivity and path-based metrics for the graph G.

    Parameters:
        G (nx.Graph or nx.DiGraph): The input graph.

    Returns:
        dict: A dictionary containing connectivity features.
    """
    features = {}

    # Handle empty graph
    if nx.is_empty(G) or G.number_of_nodes() == 0:
        features["diameter"] = 0
        features["radius"] = 0
        features["avg_path_length"] = 0
        features["num_connected_components"] = 0
        return features

    # Number of connected components
    if G.is_directed():
        features["num_weakly_connected_components"] = nx.number_weakly_connected_components(G)
        features["num_strongly_connected_components"] = nx.number_strongly_connected_components(G)
    else:
        features["num_connected_components"] = nx.number_connected_components(G)

    # Handle the largest connected component
    if G.is_directed():
        # For directed graphs, consider the largest strongly connected component
        largest_scc = max(nx.strongly_connected_components(G), key=len)
        subG = G.subgraph(largest_scc)
        if nx.is_strongly_connected(G):
            strongly_connected = True
        else:
            strongly_connected = False
    else:
        # For undirected graphs, use the largest connected component
        if nx.is_connected(G):
            subG = G
        else:
            largest_cc = max(nx.connected_components(G), key=len)
            subG = G.subgraph(largest_cc)
            strongly_connected = True  # Undirected components are inherently strongly connected

    # Diameter, radius, and average path length
    if strongly_connected and subG.number_of_nodes() > 1:
        # Diameter: the maximum eccentricity
        features["diameter"] = nx.diameter(subG)

        # Radius: the minimum eccentricity
        features["radius"] = nx.radius(subG)

        # Average shortest path length
        features["avg_path_length"] = nx.average_shortest_path_length(subG)
    else:
        # Set metrics to None or appropriate defaults if not strongly connected
        features["diameter"] = None
        features["radius"] = None
        features["avg_path_length"] = None

    return features


def graph_clustering_features(G):

    # ----------------------------------------------------------------------
    # Clustering and transitivity
    # ----------------------------------------------------------------------
    # Average clustering coefficient
    
    features = {}
    
    if G.number_of_nodes() > 1:
        features["avg_clustering"] = nx.average_clustering(G)
    else:
        features["avg_clustering"] = 0.0

    # Global clustering coefficient (transitivity)
    #  (ratio of 3 × number of triangles to number of connected triples)
    features["transitivity"] = nx.transitivity(G)

    return features


def graph_centrality_features(G):
    """
    Compute centrality features for the graph G.

    Parameters:
        G (nx.Graph or nx.DiGraph): The input graph.

    Returns:
        dict: A dictionary containing centrality measures.
    """
    features = {}

    if G.number_of_nodes() > 1:
        # Betweenness centrality
        betweenness_dict = nx.betweenness_centrality(G, normalized=True)
        features["avg_betweenness"] = np.mean(list(betweenness_dict.values()))
        features["max_betweenness"] = np.max(list(betweenness_dict.values()))

        # Closeness centrality
        closeness_dict = nx.closeness_centrality(G)
        features["avg_closeness"] = np.mean(list(closeness_dict.values()))
        features["max_closeness"] = np.max(list(closeness_dict.values()))

        # Eigenvector centrality
        try:
            eigenvector_dict = nx.eigenvector_centrality(G, max_iter=10000, tol=1e-6)
            features["avg_eigenvector"] = np.mean(list(eigenvector_dict.values()))
            features["max_eigenvector"] = np.max(list(eigenvector_dict.values()))
        except nx.NetworkXException as e:
            # Fallback if eigenvector centrality fails
            features["avg_eigenvector"] = 0
            features["max_eigenvector"] = 0
            print(f"Eigenvector centrality failed: {e}")
    else:
        # Handle graphs with 1 or fewer nodes
        features["avg_betweenness"] = 0
        features["max_betweenness"] = 0
        features["avg_closeness"] = 0
        features["max_closeness"] = 0
        features["avg_eigenvector"] = 0
        features["max_eigenvector"] = 0

    return features


def graph_structure_features(G):
    # ----------------------------------------------------------------------
    # Community structure (e.g., modularity)
    # ----------------------------------------------------------------------
    # NetworkX does not have a built-in "modularity" in core, but we can use:
    # "python-louvain" or "community" library for Louvain-based partition.
    # For example:
    
    features = {}
    
    try:
        import community  # python-louvain
        partition = community.best_partition(G)  # returns dict: node -> community_id
        # Number of communities
        num_communities = len(set(partition.values()))
        features["num_communities_louvain"] = num_communities
        
        # Convert partition dict into a list of sets for nx.algorithms.community.modularity
        communities = {}
        for node, comm_id in partition.items():
            communities.setdefault(comm_id, set()).add(node)
        communities_list = list(communities.values())
        
        # Modularity
        if len(communities_list) > 1:
            features["modularity_louvain"] = nx.algorithms.community.quality.modularity(G, communities_list)
        else:
            # If only one community is found, the modularity is not well-defined
            features["modularity_louvain"] = 0
    except ImportError:
        # If python-louvain not installed, or any error in import
        features["num_communities_louvain"] = 0
        features["modularity_louvain"] = 0

    return features


def graph_spectral_features(G):
    """
    Compute spectral properties for a graph G.

    Parameters:
        G (nx.Graph or nx.DiGraph): The input graph.

    Returns:
        dict: A dictionary containing spectral features.
    """
    features = {}

    if G.number_of_nodes() > 0:
        try:
            # Handle directed graphs by converting to undirected for Laplacian spectrum
            if G.is_directed():
                G_undirected = G.to_undirected()
            else:
                G_undirected = G

            # Laplacian spectrum (list of eigenvalues)
            lap_eigs = nx.laplacian_spectrum(G_undirected)
            lap_eigs_sorted = np.sort(lap_eigs)

            # Adjacency spectrum
            adj_eigs = nx.adjacency_spectrum(G)
            adj_eigs_sorted = np.sort(adj_eigs)

            # Algebraic connectivity: second smallest Laplacian eigenvalue
            if len(lap_eigs_sorted) >= 2:
                features["algebraic_connectivity"] = lap_eigs_sorted[1].real
            else:
                features["algebraic_connectivity"] = 0

            # Spectral radius: largest absolute eigenvalue of adjacency
            if len(adj_eigs_sorted) > 0:
                features["spectral_radius"] = max(abs(e.real) for e in adj_eigs_sorted)
            else:
                features["spectral_radius"] = 0

            # Spectral gap: difference between largest and second-largest eigenvalue
            if len(adj_eigs_sorted) >= 2:
                sorted_abs = sorted([abs(e.real) for e in adj_eigs_sorted], reverse=True)
                features["spectral_gap"] = sorted_abs[0] - sorted_abs[1]
            else:
                features["spectral_gap"] = 0

        except nx.NetworkXError as e:
            # Handle errors in spectrum computation
            features["algebraic_connectivity"] = 0
            features["spectral_radius"] = 0
            features["spectral_gap"] = 0
            features["error"] = str(e)
    else:
        # Handle empty graph case
        features["algebraic_connectivity"] = 0
        features["spectral_radius"] = 0
        features["spectral_gap"] = 0

    return features


def graph_cycles_features(G):
    """
    Compute cycle-related features for a graph G.

    Parameters:
        G (nx.Graph or nx.DiGraph): The input graph.

    Returns:
        dict: A dictionary containing cycle-related features.
    """
    features = {}

    # For cycle counts
    if G.is_directed():
        # For directed graphs, use simple_cycles to find all cycles
        cycles = list(nx.simple_cycles(G))
        features["cycle_count"] = len(cycles)
    else:
        # For undirected graphs, use cycle_basis
        cycles = nx.cycle_basis(G)
        features["cycle_count"] = len(cycles)

    # ----------------------------------------------------------------------
    # Other shape and topology descriptors
    # ----------------------------------------------------------------------
    # Assortativity (degree assortativity)
    # If the graph is smaller than 2 nodes or has no edges, this might fail
    try:
        features["degree_assortativity"] = safe_degree_assortativity_coefficient(G)
    except nx.NetworkXError:
        features["degree_assortativity"] = 0

    # Girth (length of shortest cycle)
    # For directed graphs, compute girth from simple_cycles
    if len(cycles) > 0:
        features["girth"] = min(len(cycle) for cycle in cycles)
    else:
        features["girth"] = 0

    # Planarity test
    # returns (is_planar, embedding)
    # For large graphs, this can be expensive
    try:
        if not G.is_directed():
            is_planar, _ = nx.check_planarity(G, False)
            features["is_planar"] = 1 if is_planar else 0
        else:
            # Planarity is not defined for directed graphs
            features["is_planar"] = None
    except nx.NetworkXException:
        features["is_planar"] = 0

    return features


def graph_isomorphism_features(G, max_size):
    """
    Extract more complex shape features from graph G.
    """
    features = {}
    # ----------------------------------------------------------------------
    # Isomorphism
    # ----------------------------------------------------------------------
    for i in range(3, max_size + 1):
        i_pattern = nx.cycle_graph(i)
        features[f"{i}_size_pattern_count"] = count_subgraph(G, i_pattern)
        
    return features


def graph_cliques_features(G, max_size):
    """
    Computes features related to cliques for the graph G.

    Parameters:
        G (nx.Graph or nx.DiGraph): The input graph.
        max_size (int): Maximum clique size to consider.

    Returns:
        dict: A dictionary with clique-related features.
    """
    features = {}

    # Convert directed graphs to undirected for clique analysis
    if G.is_directed():
        G_undirected = G.to_undirected()
    else:
        G_undirected = G

    # Compute maximal cliques of size k
    for i in range(3, max_size + 1):
        features[f"k{i}_count_maximal"] = count_cliques_of_size_k(G_undirected, i)

    return features

def get_graph_properties(G, max_size=5):
    """
    Extracts a variety of graph properties from a NetworkX graph.
    
    Parameters
    ----------
    G : nx.Graph
        A
    """
    features = {}
    
    # Degree-related measures
    degree_features = graph_degree_features(G)
    features.update(degree_features)
    
    # Connectivity and path-based metrics
    connectivity_features = graph_connectivity_features(G)
    features.update(connectivity_features)
    
    # Clustering and transitivity
    clustering_features = graph_clustering_features(G)
    features.update(clustering_features)
    
    # Centrality measures
    centrality_features = graph_centrality_features(G)
    features.update(centrality_features)
    
    # Community structure (e.g., modularity)
    structure_features = graph_structure_features(G)
    features.update(structure_features)
    
    # Spectral properties
    spectral_features = graph_spectral_features(G)
    features.update(spectral_features)
    
    # Cycles and other shape descriptors
    cycles_features = graph_cycles_features(G)
    features.update(cycles_features)
    
    # Isomorphism features
    isomorphism_features = graph_isomorphism_features(G, max_size)
    features.update(isomorphism_features)
    
    # Cliques features
    cliques_features = graph_cliques_features(G, max_size)
    features.update(cliques_features)
    
    return features


import networkx as nx

def graph_difference(graph, subgraph):
    """
    Computes the difference between a graph and its subgraph.
    
    Parameters:
        graph (nx.Graph or nx.DiGraph): The larger graph.
        subgraph (nx.Graph or nx.DiGraph): The subgraph to compare.
    
    Returns:
        dict: A dictionary containing:
            - "difference_nodes": Nodes in the graph but not in the subgraph.
            - "difference_edges": Edges in the graph but not in the subgraph.
            - "difference_graph": A new graph containing only the differences.
    """
    # Compute the difference in nodes
    difference_nodes = set(graph.nodes()) - set(subgraph.nodes())
    
    # Compute the difference in edges
    difference_edges = set(graph.edges()) - set(subgraph.edges())
    
    # Create a new graph containing the differences
    difference_graph = nx.DiGraph() if graph.is_directed() else nx.Graph()
    difference_graph.add_nodes_from(difference_nodes)
    
    difference_graph.add_edges_from(difference_edges)
    
    # Return the differences as a dictionary
    return difference_graph



def get_row(key, value, label, correctness, modality):
    row = {
        'Graph ID': key,
        'Graph_type': modality,
        'Classification': label,
        'Classification Outcome': True if correctness == 'correct' else False,
        'num_nodes': value['num_nodes'],
        'num_edges': value['num_edges'],
        'avg_degree': value['avg_degree'],
        'max_degree': value['max_degree'],
        'min_degree': value['min_degree'],
        'degree_variance': value['degree_variance'],
        'degree_skewness': value['degree_skewness'],
        'num_weakly_connected_components': value['num_weakly_connected_components'] if 'num_weakly_connected_components' in value else 0,
        'num_strongly_connected_components': value['num_strongly_connected_components'] if 'num_strongly_connected_components' in value else 0,
        'diameter': value['diameter'],
        'radius': value['radius'],
        'avg_path_length': value['avg_path_length'],
        'avg_clustering': value['avg_clustering'],
        'transitivity': value['transitivity'],
        'avg_betweenness': value['avg_betweenness'],
        'max_betweenness': value['max_betweenness'],
        'avg_closeness': value['avg_closeness'],
        'max_closeness': value['max_closeness'],
        'avg_eigenvector': value['avg_eigenvector'],
        'max_eigenvector': value['max_eigenvector'],
        'num_communities_louvain': value['num_communities_louvain'],
        'modularity_louvain': value['modularity_louvain'],
        'algebraic_connectivity': value['algebraic_connectivity'],
        'spectral_radius': value['spectral_radius'],
        'spectral_gap': value['spectral_gap'],
        'cycle_count': value['cycle_count'],
        'degree_assortativity': value['degree_assortativity'],
        'girth': value['girth'],
        'is_planar': value['is_planar'],
        '3_size_pattern_count': value['3_size_pattern_count'],
        '4_size_pattern_count': value['4_size_pattern_count'],
        '5_size_pattern_count': value['5_size_pattern_count'],
        'k3_count_maximal': value['k3_count_maximal'],
        'k4_count_maximal': value['k4_count_maximal'],
        'k5_count_maximal': value['k5_count_maximal'],
    }
    return row


def filter_dataframe(df, graph_type, classification, classification_outcome):
    df_filtered = df[df['Graph_type'] == graph_type]
    df_filtered = df_filtered[df_filtered['Classification Outcome'] == classification_outcome]
    df_filtered = df_filtered[df_filtered['Classification'] == classification]
    return df_filtered

def remove_columns(df):
    df = df.drop(columns=['spectral_gap', 'spectral_radius', 'girth', 'avg_clustering', 'avg_path_length', 'diameter', 'radius', 'degree_assortativity', 'algebraic_connectivity'])
    for column in df.columns:
        uniques = df[column].unique()
        if len(uniques) == 1:
            df = df.drop(columns=[column])
    return df

def generate_summaries(dataset, dataframe, DATASET_LABELS):
    summaries = {}
    for modality in ['graph', 'subgraph', 'difference_graph']:
        for classification in range(DATASET_LABELS[dataset]):
            try:
                for classification_outcome in [True, False]:
                    df_filtered = filter_dataframe(dataframe, modality, classification, classification_outcome)
                    df_filtered = remove_columns(df_filtered)
                    summaries[(modality, classification, classification_outcome)] = df_filtered.describe()
            except:
                print(f'Error in {modality}, {classification}, {classification_outcome}')
                continue
    return summaries

def generate_subdataframes(dataset, dataframe, DATASET_LABELS):
    subdataframes = {}
    for modality in ['graph', 'subgraph', 'difference_graph']:
        for classification in range(DATASET_LABELS[dataset]):
            try:
                for classification_outcome in [True, False]:
                    df_filtered = filter_dataframe(dataframe, modality, classification, classification_outcome)
                    df_filtered = remove_columns(df_filtered)
                    subdataframes[(modality, classification, classification_outcome)] = df_filtered
            except:
                print(f'Error in {modality}, {classification}, {classification_outcome}')
                continue
    return subdataframes


def save_summaries(summaries, dataset, DATASET_LABELS):
    print(f'Saving summaries for {dataset}')
    num_labels = DATASET_LABELS[dataset]
    if dataset == 'ag-news':
        real_labels = ['World', 'Sports', 'Business', 'Sci-Tech']
    elif dataset == 'sst-2':
        real_labels = ['Positive', 'Negative']
    else:
        raise ValueError('Unknown dataset')
    
    assert len(real_labels) == num_labels # Ensure the number of labels matches the dataset
    for i in range(num_labels): 
        for correctness in [True, False]:
            for type in ['graph', 'subgraph', 'difference_graph']:
                if not os.path.exists(f'usrvol/experiments/structural_properties/summaries/'):
                    os.makedirs(f'usrvol/experiments/structural_properties/summaries/', exist_ok=True)
                    summaries[(type, i, correctness)].to_csv(f'usrvol/experiments/structural_properties/summaries/summary_{real_labels[i]}_{dataset}_{type}_{correctness}.csv')
                else:
                    summaries[(type, i, correctness)].to_csv(f'usrvol/experiments/structural_properties/summaries/summary_{real_labels[i]}_{dataset}_{type}_{correctness}.csv')
def heatmap(subdataframes, figsize, dataset, DATASET_LABELS):
    num_labels = DATASET_LABELS[dataset]
    if dataset == 'ag-news':
        real_labels = ['World', 'Sports', 'Business', 'Sci-Tech']
    elif dataset == 'sst-2':
        real_labels = ['Positive', 'Negative']
    else:
        raise ValueError('Unknown dataset')
    
    assert len(real_labels) == num_labels # Ensure the number of labels matches the dataset
    for i in tqdm(range(num_labels), desc='Generating heatmaps, for dataset: ' + dataset): 
        for correctness in [True, False]:
            for type in ['graph', 'subgraph', 'difference_graph']:
                plt.figure(figsize=figsize)
                sns.heatmap(subdataframes[type, i, correctness].corr(), annot=True, cmap='magma', fmt=".2f")
                plt.title(f"Correlation Heatmap for {real_labels[i]} {type} {dataset} {'' if correctness else 'mis'}classified")
                if not os.path.exists(f"usrvol/experiments/structural_properties/heatmaps/"):
                    os.makedirs(f"usrvol/experiments/structural_properties/heatmaps/", exist_ok=True)
                plt.savefig(f"usrvol/experiments/structural_properties/heatmaps/heatmap_{real_labels[i]}_{type}_{dataset}_{'' if correctness else 'mis'}classified.pdf", 
                            format='pdf', bbox_inches='tight')
                plt.close()
    
def histograms(subdataframes, dataset, DATASET_LABELS):
    num_labels = DATASET_LABELS[dataset]
    
    if dataset == 'ag-news':
        real_labels = ['World', 'Sports', 'Business', 'Sci-Tech']
    elif dataset == 'sst-2':
        real_labels = ['Positive', 'Negative']
    else:
        raise ValueError('Unknown dataset')

    # Ensure the number of labels matches the dataset
    assert len(real_labels) == num_labels 

    # Define the output directory
    output_dir = "usrvol/experiments/structural_properties/histograms/"
    
    # Create the directory if it doesn't exist
    try:
        os.makedirs(output_dir, exist_ok=True)
    except Exception as e:
        print(f"Error creating directory {output_dir}: {e}")
        return

    for i in tqdm(range(num_labels), desc='Generating histograms, for dataset: ' + dataset): 
        for correctness in [True, False]:
            for graph_type in ['graph', 'subgraph', 'difference_graph']:
                for column in subdataframes[graph_type, i, correctness].columns:
                    try:
                        plt.figure(figsize=(10, 6))
                        # Convert boolean columns to integers if needed
                        if subdataframes[graph_type, i, correctness][column].dtype == 'bool':
                            subdataframes[graph_type, i, correctness][column] = (
                                subdataframes[graph_type, i, correctness][column].astype(int)
                            )
                        # Plot the histogram
                        subdataframes[graph_type, i, correctness][column].hist(bins=10)
                        plt.title(f"{column} for {real_labels[i]} {dataset} {graph_type} {'' if correctness else 'mis'}classified")

                        # Save the histogram
                        output_path = os.path.join(
                            output_dir, 
                            f"histogram_{real_labels[i]}_{dataset}_{graph_type}_{correctness}.pdf"
                        )
                        plt.savefig(output_path, format='pdf', bbox_inches='tight')
                        print(f"Saved histogram to {output_path}")
                        plt.close()
                    except Exception as e:
                        print(f"Error generating histogram for {graph_type}, label {i}, correctness {correctness}, column {column}: {e}")
                        
                        
                        
                        
# FOR SKIPGRAMS

def collect_tokens_from_skipgrams(skipgram_dict):
    """
    skipgram_dict is something like:
    {
      "tokenA|tokenB": countAB,
      "tokenX|tokenY": countXY,
      ...
    }
    Returns a set of all tokens that appear in any skipgram.
    """
    tokens = set()
    for skipgram in skipgram_dict:
        t1, t2 = skipgram.split("|")
        tokens.add(t1)
        tokens.add(t2)
    return tokens


def build_token_index(tokens):
    """
    tokens: set of unique tokens
    Returns a dict { token: index }, and a list of tokens in index order.
    """
    tokens = sorted(list(tokens))  # sort for consistency
    token2idx = { t: i for i, t in enumerate(tokens) }
    return token2idx, tokens


def build_cooccurrence_matrix(skipgram_dict):
    """
    skipgram_dict is e.g. merged_skipgrams[label]['correct']
    Return:
        matrix: NxN numpy array with counts
        token2idx: mapping from token to index
        idx2token: list of tokens (index -> token)
    """
    # 1. Collect tokens
    tokens = collect_tokens_from_skipgrams(skipgram_dict)
    
    # 2. Build index mapping
    token2idx, idx2token = build_token_index(tokens)
    N = len(tokens)

    # 3. Create NxN matrix of zeros
    matrix = np.zeros((N, N), dtype=np.int32)

    # 4. Fill the matrix with skipgram counts
    for skipgram, count in skipgram_dict.items():
        t1, t2 = skipgram.split("|")
        i = token2idx[t1]
        j = token2idx[t2]
        matrix[i, j] += count

    return matrix, token2idx, idx2token


def visualize_cooccurrence(matrix, idx2token, title="Co-occurrence Matrix", top_n=30):
    """
    matrix: NxN co-occurrence matrix
    idx2token: list of tokens (index -> token)
    title: optional title for the plot
    top_n: if you have many tokens, you might only plot the top N for clarity
    """
    # If you have a large number of tokens, consider slicing top_n x top_n
    # for a more readable plot. For example, let's show the top_n most frequent tokens.
    # One approach: find top_n tokens with the highest sum across the matrix rows/cols.
    
    if matrix.shape[0] > top_n:
        # sum the row occurrences to find most frequent tokens
        row_sums = matrix.sum(axis=1)
        # argsort in descending order
        top_indices = np.argsort(-row_sums)[:top_n]
        
        # filter matrix and token list
        matrix = matrix[top_indices][:, top_indices]
        idx2token = [idx2token[i] for i in top_indices]

    plt.figure(figsize=(10, 8))
    # Set 'annot=True' so each cell shows its value
    # 'fmt="g"' displays the numbers in a "general" format, good for larger counts
    sns.heatmap(matrix, 
                xticklabels=idx2token, 
                yticklabels=idx2token, 
                cmap="Blues", 
                annot=True,     # <-- display numeric values
                fmt='g')        # 'g' = general numerical format
    
    plt.title(title)
    plt.xlabel("Token (second in skipgram)")
    plt.ylabel("Token (first in skipgram)")
    plt.tight_layout()
    
    # If you're on a headless environment (like a remote server), you can remove plt.show()
    # or ignore the warnings, as the figure will still be saved.
    plt.show()

    # Save the figure as a PDF
    plt.savefig(f"/usrvol/experiments/explainability_results/{title}.pdf", 
                format='pdf', bbox_inches='tight')
    plt.close()
    
    
    
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

def visualize_cooccurrence_symmetric(matrix, idx2token, title="Co-occurrence Matrix", top_n=30):
    """
    matrix: NxN co-occurrence matrix (directional counts)
    idx2token: list of tokens (index -> token)
    title: optional title for the plot
    top_n: if you have many tokens, you might only plot the top N for clarity
    """

    # 1. Optionally slice the top-N frequent tokens
    if matrix.shape[0] > top_n:
        # sum the row occurrences to find the most frequent tokens
        row_sums = matrix.sum(axis=1)
        # sort in descending order
        top_indices = np.argsort(-row_sums)[:top_n]
        
        # filter matrix and token list
        matrix = matrix[top_indices][:, top_indices]
        idx2token = [idx2token[i] for i in top_indices]

    # 2. Make the matrix symmetric by copying values from the upper triangle to the lower triangle.
    #    For each (i, j) in the upper triangle, set (j, i) = (i, j).
    for i in range(matrix.shape[0]):
        for j in range(i + 1, matrix.shape[0]):
            matrix[j, i] = matrix[i, j]

    # 3. Plot the symmetric matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(matrix, 
                xticklabels=idx2token, 
                yticklabels=idx2token, 
                cmap="Blues", 
                annot=False,
                fmt='g')
    
    plt.title(title)
    plt.xlabel("Token (column)")
    plt.ylabel("Token (row)")
    plt.tight_layout()

    # Save to PDF (remove plt.show() if you're in a non-interactive / headless environment)
    plt.savefig(f"/usrvol/experiments/explainability_results/{title}.pdf",
                format='pdf', bbox_inches='tight')
    plt.close()


import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

def visualize_lower_triangle(matrix, idx2token, title="Lower Triangular Heatmap", top_n=30):
    """
    Plots a heatmap showing only the lower-triangular portion (including diagonal)
    of the NxN 'matrix'.

    matrix: NxN co-occurrence matrix (could be symmetric or not)
    idx2token: list of tokens (index -> token)
    title: title for the plot
    top_n: if you have many tokens, display only the top_n most frequent (by row-sum)
    """

    # 1. (Optional) Pick top_n tokens by row sum
    if matrix.shape[0] > top_n:
        row_sums = matrix.sum(axis=1)
        top_indices = np.argsort(-row_sums)[:top_n]
        matrix = matrix[top_indices][:, top_indices]
        idx2token = [idx2token[i] for i in top_indices]
    
    N = matrix.shape[0]

    # 2. Create a mask for the upper triangle.
    #    mask[i, j] = True means "hide the cell (i, j)".
    #    np.triu() by default includes the diagonal in the upper half,
    #    so we'll shift it by 1 if we want the diagonal shown.
    
    mask = np.triu(np.ones_like(matrix, dtype=bool), k=1)
    # 'k=1' => excludes the diagonal from the mask (so diagonal will be visible).
    # If you prefer to hide the diagonal as well, use k=0.

    # 3. Plot with Seaborn, applying the mask.
    plt.figure(figsize=(10, 8))
    sns.heatmap(matrix,
                mask=mask,       # Hide the upper triangle
                xticklabels=idx2token,
                yticklabels=idx2token,
                cmap="Blues",
                annot=False,
                fmt='g')         # 'g' for general number format
    
    plt.title(title)
    plt.xlabel("Token")
    plt.ylabel("Token")
    plt.tight_layout()

    # 4. Save to file and close
    plt.savefig(f"/usrvol/experiments/explainability_results/{title}.pdf",
                format='pdf', bbox_inches='tight')
    plt.close()



def trim_and_sort_matrix(matrix, idx2word, top_n=30):
    """
    1. Finds top_n words by total frequency (row + column sum).
    2. Slices the matrix to those top_n words.
    3. Sorts the sliced matrix (and word list) by descending total frequency.
    Returns:
       trimmed_matrix, trimmed_idx2word
    """
    N = len(idx2word)
    if matrix.shape[0] != N:
        raise ValueError(f"Matrix dimension {matrix.shape} doesn't match idx2word size {N}.")

    # Calculate total frequency per word (row sum + column sum)
    row_sums = matrix.sum(axis=1)
    col_sums = matrix.sum(axis=0)
    total_freq = row_sums + col_sums  # shape (N,)

    # Sort by descending frequency
    sorted_indices = np.argsort(-total_freq)  # negative => descending

    # Clamp top_n to avoid out-of-range
    top_n = min(top_n, N)
    sorted_indices = sorted_indices[:top_n]

    # Filter matrix
    trimmed_matrix = matrix[sorted_indices][:, sorted_indices]
    trimmed_idx2word = [idx2word[i] for i in sorted_indices]

    return trimmed_matrix, trimmed_idx2word


def normalize_matrix_global(matrix):
    """
    Normalize entire matrix by its global sum.
    """
    total = matrix.sum()
    if total > 0:
        return matrix / total
    return matrix  # if total=0, just return original


def plot_heatmap(matrix, idx2word, title, save_path=None, annot=False):
    """
    Plots a Seaborn heatmap for the given matrix and word labels.
    matrix: 2D numpy array
    idx2word: list of tokens (index -> token)
    title: str
    save_path: optional path to save PDF
    annot: bool, if True then numeric values will be shown in each cell
    """
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        matrix,
        xticklabels=idx2word,
        yticklabels=idx2word,
        cmap="Blues",
        annot=annot,   # <-- Use the annot argument here
        fmt=".2f" if annot else "g"  # Show decimals if annot=True, else "g"
    )
    plt.title(title)
    plt.xlabel("Second Token")
    plt.ylabel("First Token")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, format='pdf', bbox_inches='tight')
        print(f"Saved heatmap to {save_path}")
    # If headless environment, remove or ignore plt.show()
    plt.show()
    plt.close()


def plot_heatmap_log_scale(matrix, idx2word, title="Log-Scaled Co-occurrence", save_path=None, annot=False):
    """
    Plots a log-scaled heatmap so that a few very large counts
    don't overshadow the rest.
    """
    # Avoid log(0) issues: use (matrix + 1)
    base = 1.5
    matrix_log = np.log(matrix + 1) / np.log(base)
    #matrix_log = np.log2(matrix +1)  # log(1 + x)

    plt.figure(figsize=(10,8))
    sns.heatmap(
        matrix_log,
        xticklabels=idx2word,
        yticklabels=idx2word,
        cmap="Blues",
        annot=annot,
        fmt=".2f",
        # Optionally, pass a LogNorm if you want the heatmap color scale in log domain
        # norm=LogNorm(),  # comment out if you just want the data itself log-transformed
    )
    plt.title(title)
    plt.xlabel("Second Token")
    plt.ylabel("First Token")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, format='pdf', bbox_inches='tight')
        print(f"Saved heatmap to {save_path}")
    plt.show()
    plt.close()
    
def visualize_bigrams_heatmaps(merged_skipgrams, dataset, top_n=30, annot=False):
    """
    For each label and subset ('correct'/'incorrect'), build a heatmap of bigram frequencies.
    Saves each heatmap to a PDF file.
    """
    labels = sorted(merged_skipgrams.keys())  # e.g., [0, 1, 2, 3] for ag-news

    for label in labels:
        for subset in ["correct", "incorrect"]:
            skipgram_dict = merged_skipgrams[label][subset]  # { "word1|word2": freq, ... }
            
            if not skipgram_dict:
                print(f"No skipgrams for label={label}, subset={subset}. Skipping.")
                continue

            # 1. Build the raw matrix
            matrix, token2idx, idx2token = build_cooccurrence_matrix(skipgram_dict)

            # 2. Trim & sort to top_n
            matrix, idx2token = trim_and_sort_matrix(matrix, idx2token, top_n=top_n)

            # 3. Normalize globally
            matrix = normalize_matrix_global(matrix)

            # 4. Construct a descriptive title
            label_str = LABEL_MAPPER[dataset][label]  # e.g. "World", "Positive", etc.
            title = f"Co-occurrence Matrix for {dataset}, label={label_str} ({subset})"

            # 5. Plot heatmap
            save_path = f"/usrvol/experiments/explainability_results/{dataset}_label{label}_{subset}_heatmap.pdf"
            plot_heatmap_log_scale(matrix, idx2token, title, save_path=save_path, annot=annot)


