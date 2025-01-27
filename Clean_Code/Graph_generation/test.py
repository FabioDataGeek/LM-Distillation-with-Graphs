import networkx as nx
from sintactic import *
from semantic import *
from constituency import *
from hierarchical_wn import *
from mixer import *


TEST_TYPE_DICT = {
    'sintactic': sintactic_graph_generator,
    'semantic': semantic_graph_generator,
    'constituency': constituency_graph_generator,
    #'knowledge': 'NOT IMPLEMENTED'

}

def test(test_type, sentence, model):
    tokens = sentence.split()

    # change this class for every check
    generator = TEST_TYPE_DICT[test_type](model=model)    

    # Check if every method is working correctly
    graph = generator.get_graph(tokens)
    generator.draw_graph(graph)
    generator.save_graph(graph, '/usrvol/test/graphs', f"{test_type}")


""" sentence = "I saw Sarah with a telescope."
tokens = sentence.split()

# change this class for every check
generator = semantic_graph_generator(model='dep-biaffine-roberta-en')    

# Check if every method is working correctly
graph = generator.get_graph(tokens)
generator.draw_graph(graph)
generator.save_graph(graph, 'usrvol/test/graphs', f"{test_type}.pkl") """

""" mixer = GraphMixer()
sintactic = nx.read_gpickle('usrvol/test/graphs/sintactic.pkl')
semantic = nx.read_gpickle('usrvol/test/graphs/semantic.pkl')
constituency = nx.read_gpickle('usrvol/test/graphs/constituency.pkl')
#knowledge = nx.read_gpickle('usrvol/test/graphs/knowledge.pkl') # NOT IMPLEMENTED
graph = mixer.mix([sintactic, semantic, constituency], sintactic=True, semantic=True, constituency=True)
nx.draw_kamada_kawai(graph, with_labels=True)
plt.show()
nx.write_gpickle(graph, 'usrvol/test/graphs/mixed.pkl') """


""" models:
sintactic: 'dep-biaffine-roberta-en'
semantic: 'sdp-vi-en'
constituency: 'con-crf-roberta-en'
 """

if __name__ == '__main__':
    test('sintactic', "I saw Sarah with a telescope.", 'dep-biaffine-roberta-en')
    #test('semantic', "I saw Sarah with a telescope.", 'sdp-vi-en')
    #test('constituency', "I saw Sarah with a telescope.", 'con-crf-roberta-en')
    #test('knowledge', "I saw Sarah with a telescope.") # NOT IMPLEMENTED
    print("All tests passed!")