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

def test_graph_sparsity(n, p, viz=False):
    G = nx.complete_graph(n)
    G_ugly = ugly_graph(G, p)
    if viz:
        pos = nx.spring_layout(G_ugly)
        nx.draw(G_ugly, pos, with_labels=True, node_color='green', edge_color='black')
        plt.title(f"uglied graph on {n} vertices")
        plt.show()
    return G_ugly

# extremal examples

def join_graphs(G1, H):
    G = nx.union(G1, H)
    for u in G1.nodes():
        for v in H.nodes():
            G.add_edge(u, v)
    return G

def disjoint_union(graphs):
    H = nx.Graph()
    for G in graphs:
        H = nx.union(H, G)
    return H

def invert_mapping(d):
    inv = {}
    for key, color in d.items():
        inv.setdefault(color, []).append(key)
    return inv

def N_k_n(k, n):
    if n < 2*k:
        raise ValueError("n must be at least 2*k")
    G1 = nx.complete_graph(k)
    G1 = nx.relabel_nodes(G1, lambda i: ('G1', i))
    num_H1 = n - 2*k
    H1 = nx.complete_graph(num_H1)
    H1 = nx.relabel_nodes(H1, lambda i: ('H1', i))
    H2 = nx.empty_graph(k)
    H2 = nx.relabel_nodes(H2, lambda i: ('H2', i))
    H = disjoint_union([H1, H2])
    G = join_graphs(G1, H)
    
    vertex_colors = {}
    for node in G.nodes():
        group, _ = node
        if group == 'G1':
            vertex_colors[node] = "blue"
        elif group == 'H1':
            vertex_colors[node] = "gray"
        elif group == 'H2':
            vertex_colors[node] = "black"
        else:
            vertex_colors[node] = "unknown"
    
    edge_colors = {}
    for u, v in G.edges():
        c1 = vertex_colors[u]
        c2 = vertex_colors[v]
        if c1 == c2:
            if c1 == "gray":
                edge_colors[(u, v)] = 'pink'
            else:
                edge_colors[(u, v)] = 'green'
        else:
            edge_colors[(u, v)] = 'gray'
    
    return G#, vertex_colors, edge_colors


def L_k_n(k, n):
    G1 = nx.empty_graph(1)
    G1 = nx.relabel_nodes(G1, lambda i: ('G1', i))

    H1 = nx.complete_graph(k)
    H1 = nx.relabel_nodes(H1, lambda i: ('H1', i))
    
    H2 = nx.complete_graph(n - k - 1)
    H2 = nx.relabel_nodes(H2, lambda i: ('H2', i))
    
    H = disjoint_union([H1, H2])
    
    G = join_graphs(G1, H)
    
    vertex_colors = {}
    for node in G.nodes():
        group, _ = node
        if group == 'G1':
            vertex_colors[node] = 'pink'
        elif group == 'H1':
            vertex_colors[node] = 'black'
        elif group == 'H2':
            vertex_colors[node] = 'gray'
        else:
            vertex_colors[node] = 'unknown'
    
    edge_colors = {}
    for u, v in G.edges():
        c1 = vertex_colors[u]
        c2 = vertex_colors[v]
        if c1 == c2:
            if c1 == 'black':
                edge_colors[(u, v)] = 'black'
            else:
                edge_colors[(u, v)] = 'gray'
        else:
            edge_colors[(u, v)] = 'red'
    
    return G #, vertex_colors, edge_colors

def L_k_n_bar(k, n):
    H1 = nx.complete_graph(k)
    H1 = nx.relabel_nodes(H1, lambda i: ('H1', i))
    
    H2 = nx.complete_graph(n - k - 1)
    H2 = nx.relabel_nodes(H2, lambda i: ('H2', i))
    
    G = disjoint_union([H1, H2])
    
    vertex_colors = {}
    for node in G.nodes():
        group, _ = node
        if group == 'H1':
            vertex_colors[node] = 'black'
        elif group == 'H2':
            vertex_colors[node] = 'gray'
        else:
            vertex_colors[node] = 'unknown'
    
    edge_colors = {}
    for u, v in G.edges():
        c1 = vertex_colors[u]
        c2 = vertex_colors[v]
        if c1 == c2:
            if c1 == 'black':
                edge_colors[(u, v)] = 'black'
            else:
                edge_colors[(u, v)] = 'gray'
        else:
            edge_colors[(u, v)] = 'red'
    
    return G #, vertex_colors, edge_colors

def N_k_n_bar(k, n):
    if n < 2*k + 1:
        raise ValueError("n must be at least 2*k+1")

    G1 = nx.complete_graph(k)
    G1 = nx.relabel_nodes(G1, lambda i: ('G1', i))

    num_H1 = n - 2*k - 1
    H1 = nx.complete_graph(num_H1)
    H1 = nx.relabel_nodes(H1, lambda i: ('H1', i))

    H2 = nx.empty_graph(k+1)
    H2 = nx.relabel_nodes(H2, lambda i: ('H2', i))

    H = disjoint_union([H1, H2])

    G = join_graphs(G1, H)
    
    vertex_colors = {}
    for node in G.nodes():
        group, _ = node
        if group == 'G1':
            vertex_colors[node] = "blue"
        elif group == 'H1':
            vertex_colors[node] = "gray"
        elif group == 'H2':
            vertex_colors[node] = "black"
        else:
            vertex_colors[node] = "unknown"
    
    edge_colors = {}
    for u, v in G.edges():
        c1 = vertex_colors[u]
        c2 = vertex_colors[v]
        if c1 == c2:
            if c1 == "gray":
                edge_colors[(u, v)] = 'pink'
            else:
                edge_colors[(u, v)] = 'green'
        else:
            edge_colors[(u, v)] = 'gray'
    
    return G #, vertex_colors, edge_colors


def generate_classic_nonhamiltonian_graphs(n):
    graphs = {}
    graphs["Petersen"] = nx.petersen_graph()
    k = np.random.randint(n)
    try:
        graphs[f"N_k_n (k=2, n={n})"] = N_k_n(2, n)
    except Exception as e:
        print("Error generating N_k_n:", e)
    
    try:
        graphs[f"L_k_n (k=3, n={n})"] = L_k_n(3, n)
    except Exception as e:
        print("Error generating L_k_n:", e)
        
    try:
        graphs[f"L_k_n_bar (k=3, n={8})"] = L_k_n_bar(3, n)
    except Exception as e:
        print("Error generating L_k_n_bar:", e)
        
    try:
        graphs[f"N_k_n_bar (k=2, n={n})"] = N_k_n_bar(2, n)
    except Exception as e:
        print("Error generating N_k_n_bar:", e)
    
    return graphs