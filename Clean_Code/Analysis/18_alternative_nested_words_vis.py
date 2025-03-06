import matplotlib.pyplot as plt
import numpy as np
import pickle as pkl
import networkx as nx
import re
from networkx.drawing.nx_agraph import graphviz_layout
from wordcloud import WordCloud
from matplotlib.offsetbox import OffsetImage, AnnotationBbox

def sanitize(text):
    """Replace problematic characters in text for labels/attributes."""
    if isinstance(text, str):
        # Replace colon with hyphen, remove guillemets and quotes.
        text = text.replace(":", "-").replace("«", "").replace("»", "").replace('"', "")
        return text
    return text

def safe_id(text):
    """Generate a safe string for node IDs: remove spaces and punctuation."""
    if isinstance(text, str):
        safe = text.replace(" ", "_")
        safe = re.sub(r'[^\w_]', '', safe)
        return safe
    return str(text)

def add_to_trie(key, words, trie):
    """Insert the given key (tuple of phrases) and associated words into the trie."""
    node = trie
    for phrase in key:
        clean_phrase = sanitize(phrase)
        if clean_phrase not in node:
            node[clean_phrase] = {"_words": [], "_children": {}}
        node[clean_phrase]["_words"].extend(words)
        node = node[clean_phrase]["_children"]

def add_nodes_from_trie(parent, subtree, path, G):
    """
    Recursively add nodes from the trie into a NetworkX DiGraph.
    Each node gets an attribute 'full_key' which is the sequence of phrases (joined with ' -> ').
    """
    for phrase, info in subtree.items():
        new_path = path + (phrase,)
        node_id = "ROOT_" + "_".join(safe_id(phrase) for phrase in new_path)
        # Concatenate words and sanitize; truncate to 200 characters if needed.
        words_text = sanitize(" ".join(info["_words"]))[:200]
        clean_label = sanitize(phrase)
        full_key = " -> ".join(new_path)
        G.add_node(node_id, label=clean_label, words=words_text, full_key=full_key)
        G.add_edge(parent, node_id)
        add_nodes_from_trie(node_id, info["_children"], new_path, G)

# Load your data (replace with your actual nested dictionary)
dataset = "ag-news"
with open(f"/usrvol/experiments/explainability_results/{dataset}_nested_words.pkl", "rb") as f:
    data = pkl.load(f)

# Loop over each combination of dataset label and correctness.
for label in data:
    for correctness in data[label]:
        # Build order_to_words for the specific combination.
        order_to_words = {}
        for subgraph in data[label][correctness]:
            for word, phrases in subgraph.items():
                key = tuple(phrases)  # preserve order
                order_to_words.setdefault(key, []).append(word)
        
        # Filter to top 10 unique phrase orders (by number of words)
        top_n = 10
        sorted_items = sorted(order_to_words.items(), key=lambda x: len(x[1]), reverse=True)[:top_n]
        order_to_words_filtered = dict(sorted_items)
        
        # Build a trie from the filtered keys.
        trie = {}
        for key, words in order_to_words_filtered.items():
            clean_key = tuple(sanitize(phrase) for phrase in key)
            add_to_trie(clean_key, words, trie)
        
        # Convert the trie into a NetworkX DiGraph.
        G = nx.DiGraph()
        G.add_node("ROOT", label="ROOT", words="", full_key="ROOT")
        add_nodes_from_trie("ROOT", trie, (), G)
        
        # Compute layout using Graphviz (specify the root).
        pos = graphviz_layout(G, prog='dot', root='ROOT')
        fig, ax = plt.subplots(figsize=(12, 12))
        nx.draw_networkx_edges(G, pos, arrows=True, ax=ax)
        
        # Overlay word cloud images on each node (except ROOT)
        for node, (x, y) in pos.items():
            if node == "ROOT":
                continue
            words_text = G.nodes[node].get('words', "")
            if words_text.strip() == "":
                continue
            try:
                wc = WordCloud(width=150, height=100, background_color='white').generate(words_text)
            except ValueError:
                continue
            wc_image = wc.to_array()
            oi = OffsetImage(wc_image, zoom=0.6)
            ab = AnnotationBbox(oi, (x, y), frameon=False, pad=0.2)
            ax.add_artist(ab)
            # Retrieve the full key attribute and add as text above the word cloud.
            full_key = G.nodes[node].get('full_key', "")
            # Adjust the vertical offset as needed.
            ax.text(x, y + 30, full_key, fontsize=8, ha='center', va='bottom', color='black')
        
        ax.set_axis_off()
        plt.title(f"Visualization of Top {top_n} Unique Phrase Orders as a Trie\nLabel: {label}, Correctness: {correctness}")
        plt.tight_layout()
        if label == "Sci/Tech":
            output_label = "SciTech"
        else:
            output_label = label
        output_filename = f"{dataset}_{output_label}_{correctness}_phrase_trie.pdf"
        plt.savefig(output_filename, format='pdf')
        plt.close()
        print(f"Saved plot for Label: {label}, Correctness: {correctness} as {output_filename}")
