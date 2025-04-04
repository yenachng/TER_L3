import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import preprocessing


def ugly_graph(G, removal_prob=0.7):
    H = G.copy()
    edges = list(H.edges())
    for u,v in edges:
        if np.random.rand() < removal_prob:
            H.remove_edge(u,v)
    padded_H = preprocessing.pad_pendents(H)
    return padded_H

def test_graph(n):
    G = nx.complete_graph(n)
    G_ugly = ugly_graph(G)
    #pos = nx.spring_layout(G_ugly)
    #nx.draw(G_ugly, pos, with_labels=True, node_color='green', edge_color='black')
    #plt.title(f"uglied graph on {n} vertices")
    #plt.show()
    return G_ugly

def test_graph_sparsity(n, p):
    G = nx.complete_graph(n)
    G_ugly = ugly_graph(G, p)
    #pos = nx.spring_layout(G_ugly)
    #nx.draw(G_ugly, pos, with_labels=True, node_color='green', edge_color='black')
    #plt.title(f"uglied graph on {n} vertices")
    #plt.show()
    return G_ugly