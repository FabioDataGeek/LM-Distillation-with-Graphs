from supar import Parser
import networkx as nx
import matplotlib.pyplot as plt
import pickle as pkl
import torch

class semantic_graph_generator():

    """
    This class is used to create a semantic graph from a sentence using a semantic parser.
    """


    def __init__(self, model: str, device: str = 'cuda:0'):
        
        """
        Initialize the semantic parser and set the device.

        Parameters:
        model (str): The model to be loaded into the semantic parser.
        device (str): The device to be used for computations ('cpu' or 'gpu'). Default is 'cpu'.
        """
        self.model = model
        self.property = 'semantic'
        self.sem = Parser.load(model)
        self.device = device
        torch.cuda.set_device(device)

    
    def _parse(self, sentence: list):

        """
        Parse the sentence using the semantic parser.
        """

        return self.sem.predict(sentence, verbose=False)[0]
    
    def _build_graph(self, semantic_tree):

        """
        Build a graph from the parsed tree.
        """
        graph = nx.DiGraph()
        
        for i in range(len(semantic_tree.values[1])):
            graph.add_node(i+1, word=semantic_tree.values[1][i])

        if len(semantic_tree.values[8]) == len(semantic_tree.values[1])-1:
            semantic_tree.values[8].append('punct')

        for i in range(len(semantic_tree.values[8])):
            parents = semantic_tree.values[8][i]
            if '_' in parents or 'punct' in parents:
                continue
            if '|' in parents:
                parents = parents.split('|')
                for parent in parents:
                    if parent[0] == '0':
                        continue
                    par, relation = int(parent.split(':')[0]), parent.split(':')[1]
                    child = int(semantic_tree.values[0][i])
                    graph.add_edge(par, child, label=relation)
            else:
                if parents[0] == '0':
                        continue    
                parents, relation = int(parents.split(':')[0]), parents.split(':')[1]
                child = int(semantic_tree.values[0][i])
                graph.add_edge(parents, child, label=relation)

        graph.graph['model'] = self.model
        graph.graph['property'] = self.property
        if id:
            graph.graph['id'] = id 
        return graph

    def get_graph(self, sentence: list):
        
        """
        Get the semantic graph of a sentence.
        """

        semantic_tree = self._parse(sentence)
        graph = self._build_graph(semantic_tree)
        return graph
    
    def draw_graph(self, graph):
        
        """
        Draw the syntactic graph.
        """

        labels = nx.get_node_attributes(graph, 'label')
        nx.draw_kamada_kawai(graph, with_labels=True, font_weight='bold')
        plt.show()

    def save_graph(self, graph, folder, filename):
        
        """
        Save the syntactic graph to a file.
        """

        with open(f'{folder}/{filename}.pkl', 'wb') as f:
            pkl.dump(graph, f)
    
