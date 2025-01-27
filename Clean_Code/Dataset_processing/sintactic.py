from supar import Parser
import networkx as nx
import matplotlib.pyplot as plt
import pickle as pkl
import torch

class sintactic_graph_generator():

    """
    This class is used to create a syntactic graph from a sentence. It first parses the sentence using a syntactic parser, 
    then builds a graph from the parsed tree. The graph can be drawn or saved to a file. The _parse, _build_graph, methods 
    are helper methods used by the get_graph method. 

    Notice that ONLY THREE METHODS SHOULD BE USED IN THIS CLASS: get_graph, draw_graph, and save_graph.
    """

    def __init__(self, model: str, device: str = 'cuda:0'):
        
        """
        Initialize the syntactic parser with a given model.
        """
        self.model = model
        self.property = 'sintactic'
        self.sint = Parser.load(model)    # ['dep-biaffine-roberta-en']
        self.device = device
        torch.cuda.set_device(device)

    def _parse(self, sentences: list[str]):
        """
        Parse the sentences using the syntactic parser.
        """
        return self.sint.predict(sentences, verbose=False, lang='en')
    
    def _build_graph(self, trees, id = False):

        """
        Build a graph from the parsed tree.
        """
        graph_list = []
        for tree in trees:
            graph = nx.DiGraph()
            for i in range(len(tree.values[1])):
                graph.add_node(i+1, label=tree.values[1][i])
            for i in range(len(tree.values[6])):
                parent = int(tree.values[6][i])
                if parent == 0:
                    continue
                else:
                    child = int(tree.values[0][i])
                    graph.add_edge(parent, child, label=tree.values[7][i])
            
            # General information of the graph i.e model used for obtain it and type of graph
            graph.graph['model'] = self.model
            graph.graph['property'] = self.property
            if id:
                graph.graph['id'] = id
            graph_list.append(graph)


        return graph_list
    
    def get_graph(self, sentence: list):
        
        """
        Get the syntactic graph of a sentence.
        """
        # for a list of sentences
        sintactic_tree = self._parse(sentence)
        graphs = self._build_graph(sintactic_tree)
        return graphs
    
    def draw_graph(self, graph):
        
        """
        Draw the syntactic graph.
        """

        labels = nx.get_node_attributes(graph, 'label')
        nx.draw_kamada_kawai(graph, with_labels=True,  font_weight='bold')
        plt.show()

    def save_graph(self, graph, folder, filename):
        
        """
        Save the syntactic graph to a file.
        """

        with open(f'{folder}/{filename}.pkl', 'wb') as f:
            pkl.dump(graph, f)