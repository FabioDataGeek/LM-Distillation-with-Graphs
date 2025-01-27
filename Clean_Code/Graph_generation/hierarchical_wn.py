import nltk
from nltk.corpus import wordnet as wn
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import re
import networkx as nx

nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')

# Get the minimum hierarchies between pairs of words in a given sentence
class wordnet_hierarchies():
    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))

    def lemmatize_sentence(self, sentence):
        pattern = '[^A-Za-z ]+'  # pattern to match non-alphabetic characters
        sentence = re.sub(pattern, '', sentence)  # substitute matched pattern with ''
        lemmatizer = self.lemmatizer
        stop_words = set(stopwords.words('english'))
        tokenized_words = word_tokenize(sentence)
        lemmas = [lemmatizer.lemmatize(word) for word in tokenized_words if word not in stop_words]
        return lemmas

    def get_synsets(self, lemmas):
        synsets = []
        for lemma in lemmas:
            synsets.append([])
            synsets[-1].append(wn.synsets(lemma))
        return synsets

    # Obtiene el camino más corto entre dos palabras en base a todos los synsets de ambas
    def shortest_path(self, synsets1, synsets2):
        shortest_path = float('inf')
        couple = (None, None)
        for synset1 in synsets1:
            for synset2 in synsets2:
                path = synset1.shortest_path_distance(synset2)
                if path is not None and path < shortest_path:
                    shortest_path = path
                    couple = (synset1, synset2)
        return shortest_path, couple
        
    # Función para obtener el camino más corto entre todos los pares de synsets de la oración
    def sentence_sense_paths(self, synsets):
        paths = []
        for i in range(len(synsets)):
            for j in range(i+1, len(synsets)):
                paths.append(self.shortest_path(synsets[i][0], synsets[j][0]))
        new_paths = []
        for path in paths:
            if path[1][0] is not None:
                new_paths.append(path)
        return new_paths

    # Esta función obtiene el synset jerárquico común más bajo entre dos synsets, 
    # así como los caminos jerárquicos de ambos
    def find_synsets_between(self, synset1, synset2):
        lch = synset1.lowest_common_hypernyms(synset2)
        hypernym_path1 = synset1.hypernym_paths()[0]
        hypernym_path2 = synset2.hypernym_paths()[0]
        return lch, hypernym_path1, hypernym_path2

    # Esta función nos devuelve, para una lista que representa el árbol jerárquico de un synset,
    # un árbol reducido en base al hiperónimo común que hemos determinado antes.
    def reduced_hierarchies(self, common_hypernym, hypernym_path):
        index = hypernym_path.index(common_hypernym)
        synsets = hypernym_path[index:]
        new_hierarchy = []
        for synset in synsets:
            new_hierarchy.append(synset)
        return new_hierarchy

    def total_hierarchies(self, sentence):
        lemmas = self.lemmatize_sentence(sentence)
        synsets = self.get_synsets(lemmas)
        synsets_pairs = self.sentence_sense_paths(synsets)
        hierarchies = []
        for pair in synsets_pairs:
            synsets_between = self.find_synsets_between(pair[1][0], pair[1][1])
            hie1 = self.reduced_hierarchies(synsets_between[0][0], synsets_between[1])
            hie2 = self.reduced_hierarchies(synsets_between[0][0], synsets_between[2])
            hierarchies.append(hie1)
            hierarchies.append(hie2)
        return hierarchies


# Construct graphs based on the hierarchies obtained from the wordnet_hierarchies class
# It is intended to build a graph for each sentence
class hierarchical_graph():
    def __init__(self):
        self.graph = nx.DiGraph()

    def add_hierarchy(self, hierarchy):
        for i in range(len(hierarchy)-1):
            self.graph.add_edge(hierarchy[i], hierarchy[i+1])

    def add_hierarchies(self, hierarchies):
        for hierarchy in hierarchies:
            self.add_hierarchy(hierarchy)

    def draw_graph(self):
        nx.draw(self.graph, with_labels=True, font_weight='bold')

    def get_graph(self):
        return self.graph

    def get_roots(self):
        roots = []
        for node in self.graph.nodes:
            if len(list(self.graph.predecessors(node))) == 0:
                roots.append(node)
        return roots

    def get_leaves(self):
        leaves = []
        for node in self.graph.nodes:
            if len(list(self.graph.successors(node))) == 0:
                leaves.append(node)
        return leaves

    def get_paths(self, node1, node2):
        return list(nx.all_simple_paths(self.graph, source=node1, target=node2))

    def get_shortest_path(self, node1, node2):
        return nx.shortest_path(self.graph, source=node1, target=node2)

    def get_shortest_path_length(self, node1, node2):
        return nx.shortest_path_length(self.graph, source=node1, target=node2)

    def get_common_hypernyms(self, node1, node2):
        return list(nx.lowest_common_ancestors(self.graph, node1, node2))

    def get_common_hypernyms_paths(self, node1, node2):
        common_hypernyms = self.get_common_hypernyms(node1, node2)
        paths = []
        for hypernym in common_hypernyms:
            paths.append(nx.shortest_path(self.graph, source=node1, target=hypernym))
            paths.append(nx.shortest_path(self.graph, source=node2, target=hypernym))
        return paths

    def get_common_hypernyms_length(self, node1, node2):
        common_hypernyms = self.get_common_hypernyms(node1, node2)
        return len(common_hypernyms)

    def get_common_hypernyms_length(self, node1, node2):
        common_hypernyms = self.get_common_hypernyms(node1, node2)
        return len(common_hypernyms)
