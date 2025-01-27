import networkx as nx


class GraphMixer():
    """
    This class will mix different types of graphs i.e. syntactic, semantic, constituency or
    knowledge graphs into a single graph based on the user's choice and the individual graphs provided.    
    """
    def __init__(self):   # a list of networkx graphs
        """
        Initialize the mixer with the graphs to be mixed.
        """

    def mix(self, graph_list:str, sintactic=False, semantic=False, constituency=False, knowledge=False):
        """
        Mix the graphs based on the user's choice.
        """
        """ id = graph_list[0].graph['id']

        for graph in graph_list:
            assert graph.graph['id'] == id, "The graphs must have the same id." """
        
        # Check the types of graphs to be mixed
        types = []
        sanity_check = []
        if sintactic:
            types.append('sintactic')
            sanity_check.append(0)
        if semantic:
            types.append('semantic')
            sanity_check.append(0)
        if constituency:
            types.append('constituency')
            sanity_check.append(0)
        if knowledge:
            types.append('knowledge')
            sanity_check.append(0)

        G = nx.MultiDiGraph()
        G.graph['property'] = []
        G.graph['model'] = []
        G.graph['id'] = id
        prop_list = []
        model_list = []
        for graph in graph_list:
            graph = nx.MultiDiGraph(graph)
            # make sure the graph used is in the types that we want
            if graph.graph['property'] in types:
                prop = graph.graph['property']
                if 'property' in G.graph:
                    # Only one graph per each type are allowed
                    if sanity_check[types.index(graph.graph['property'])] > 1:
                        raise ValueError(f"There are more than one {prop} graph.")
                    else:
                        prop_list.append(prop)
                        model_list.append(graph.graph['model'])
                        G = nx.compose(G, graph)
                        sanity_check[types.index(graph.graph['property'])] += 1
        G.graph['property'] = prop_list
        G.graph['model'] = model_list
        return G
                
""" import pickle as pkl

graph_list = []

with open('/usrvol/test/graphs/sintactic.pkl', 'rb') as f:
    graph_list.append(pkl.load(f))

with open('/usrvol/test/graphs/semantic.pkl', 'rb') as f:
    graph_list.append(pkl.load(f))

with open('/usrvol/test/graphs/constituency.pkl', 'rb') as f:
    graph_list.append(pkl.load(f))

mixer =GraphMixer()
merged_graph = mixer.mix(graph_list, sintactic=True, semantic=True, constituency=True) """