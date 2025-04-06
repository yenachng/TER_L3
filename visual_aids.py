import networkx as nx
import matplotlib.pyplot as plt
from cluster_refinement import kelmans_op, bondy_chvatal_closure
from itertools import combinations



def draw(G):
    pos = nx.kamada_kawai_layout(G)
    nx.draw(G, pos, with_labels=True, node_color='lightblue', edge_color='gray')
    plt.title(f"{G}")
    plt.show()

def largest_clique(G):
    cliques = list(nx.find_cliques(G))
    if not cliques:
        return set()
    max_size = max(len(c) for c in cliques)
    for clique in cliques:
        if len(clique) == max_size:
            return set(clique)

def draw_original_vs_closed(G):
    closedG = bondy_chvatal_closure(G)
    pos_orig = nx.kamada_kawai_layout(G)
    pos_closed = nx.kamada_kawai_layout(closedG)
    clique = largest_clique(closedG)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 7))
    axes[0].set_title(f"G: {G.number_of_edges()} edges")
    nx.draw(G, pos_orig, with_labels=True, node_color='lightblue', edge_color='gray', ax=axes[0])
    
    axes[1].set_title(f"cl(G): {closedG.number_of_edges()} edges")
    nx.draw(closedG, pos_closed, with_labels=True, node_color='lightblue', edge_color='gray', ax=axes[1])
    if clique:
        nx.draw_networkx_nodes(closedG, pos_closed, nodelist=list(clique),
                               node_color='none', edgecolors='red', linewidths=2, ax=axes[1])
    
    plt.tight_layout()
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


def test_modifications(G):
    groups = {"no changes": [], "isomorphic": [], "has_pendent_edge": [], "has_weak_edge": [], "is_disconnected": [], "other": []}
    for u, v in combinations(G.nodes(), 2):
        candidate = (u, v) if G.degree(u) > G.degree(v) else (v, u)
        H = kelmans_op(G, candidate[0], candidate[1])
        if edges_equal(G, H):
            groups["no changes"].append(candidate)
        elif nx.is_isomorphic(G, H):
            groups["isomorphic"].append(candidate)
        elif any(deg == 1 for _, deg in H.degree()):
            groups["has_pendent_edge"].append(candidate)
        elif list(nx.bridges(H)):
            groups["has_weak_edge"].append(candidate)
        elif not nx.is_connected(H):
            groups["is_disconnected"].append(candidate)
        else:
            groups["other"].append(candidate)
    for group, candidates in groups.items():
        if group == "has_pendent_edge" or group == "has_weak_edge" or group=="is disconnected":
            for candidate in candidates:
                u,v = candidate
                print(f"shift {u} -> {v}, modification type: {group}")
                visualize_kelmans_operation(G, candidate[0], candidate[1])
        else:
            print(f"{group}: {candidates}")
# need to add large graph drawing fct