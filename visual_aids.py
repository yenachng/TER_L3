import networkx as nx
import matplotlib.pyplot as plt
from cluster_refinement import kelmans_op


def draw(G):
    pos = nx.kamada_kawai_layout(G)
    nx.draw(G, pos, with_labels=True, node_color='lightblue', edge_color='black')
    plt.title(f"{G}")
    plt.show()

def visualize_kelmans_operation(G, u, v):
    H = kelmans_op(G, u, v)
    shiftable_edges = [(u, w) for w in G.neighbors(u) if w != v and w not in set(G.neighbors(v))]
    non_shifted_edges = [(u, w) for w in G.neighbors(u) if w != v and (u, w) not in shiftable_edges]
    if G.has_edge(u, v):
        non_shifted_edges.append((u, v))
    normal_edges = [e for e in G.edges() if u not in e]
    
    new_edges = [(v, w) for w in H.neighbors(v) if not G.has_edge(v, w) and not G.has_edge(w, v)]
    H_normal_edges = [e for e in H.edges() if e not in new_edges and (e[1], e[0]) not in new_edges]
    
    pos_original = nx.kamada_kawai_layout(G)
    pos_after = nx.kamada_kawai_layout(H)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    ax = axes[0]
    ax.set_title("original")
    node_colors = ['red' if n in [u, v] else 'lightblue' for n in G.nodes()]
    nx.draw_networkx_nodes(G, pos_original, node_color=node_colors, ax=ax)
    nx.draw_networkx_labels(G, pos_original, ax=ax)
    nx.draw_networkx_edges(G, pos_original, edgelist=normal_edges, edge_color='gray', ax=ax)
    nx.draw_networkx_edges(G, pos_original, edgelist=non_shifted_edges, edge_color='blue', ax=ax)
    nx.draw_networkx_edges(G, pos_original, edgelist=shiftable_edges, edge_color='orange', style='dashed', ax=ax)
    
    ax = axes[1]
    ax.set_title("after kelmans op")
    node_colors_H = ['red' if n in [u, v] else 'lightgreen' for n in H.nodes()]
    nx.draw_networkx_nodes(H, pos_after, node_color=node_colors_H, ax=ax)
    nx.draw_networkx_labels(H, pos_after, ax=ax)
    nx.draw_networkx_edges(H, pos_after, edgelist=H_normal_edges, edge_color='gray', ax=ax)
    nx.draw_networkx_edges(H, pos_after, edgelist=new_edges, edge_color='green', width=2, ax=ax)
    
    plt.tight_layout()
    plt.show()

