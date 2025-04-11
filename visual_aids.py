import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from cluster_refinement import bondy_chvatal_closure, cheeger_cut
import spectral_embedding as se

def draw_graph(g, title=None, node_color='lightblue', edge_color='gray'):
    pos = nx.kamada_kawai_layout(g)
    nx.draw(g, pos, with_labels=True, node_color=node_color, edge_color=edge_color)
    if title is None:
        title = g.graph.get("name", f"graph: |v|={g.number_of_nodes()}, |e|={g.number_of_edges()}")
    plt.title(title)
    plt.axis('off')
    plt.show()

def draw_graph_summary(g):
    if "name" in g.graph:
        title = f"{g.graph['name']} (|e|={g.number_of_edges()})"
    else:
        h, _ = cheeger_cut(g)
        rho = se.spectral_radius(g)
        title = f"cheeger: {h:.4f} | spectral rad: {rho:.4f} | |e|={g.number_of_edges()}"
    draw_graph(g, title=title)

def draw_original_vs_closed(g, title=None):
    closed_g = bondy_chvatal_closure(g)
    pos_orig = nx.kamada_kawai_layout(g)
    pos_closed = nx.kamada_kawai_layout(closed_g)
    clique = se.largest_clique(closed_g)
    if title is None:
        title = g.graph.get("name", f"|v|={g.number_of_nodes()}, |e|={g.number_of_edges()}")
    fig, axes = plt.subplots(1, 2, figsize=(14, 7))
    axes[0].set_title(f"edgecount: {g.number_of_edges()}")
    nx.draw(g, pos_orig, with_labels=False, node_color='lightblue', edge_color='black', ax=axes[0])
    axes[1].set_title(f"edgecount: {closed_g.number_of_edges()}")
    nx.draw(closed_g, pos_closed, with_labels=False, node_color='lightblue', edge_color='black', ax=axes[1])
    if clique:
        nx.draw_networkx_nodes(closed_g, pos_closed, nodelist=list(clique),
                               node_color='none', edgecolors='blue', linewidths=2, ax=axes[1])
    plt.suptitle(title)
    plt.tight_layout()
    plt.show()

def draw_comparison(G, H, nameG=None, nameH=None):
    pos1 = nx.kamada_kawai_layout(G)
    pos2 = nx.kamada_kawai_layout(H)
    fig, axes = plt.subplots(1, 2, figsize=(14, 7))
    hG, _ = cheeger_cut(G)
    rhoG = se.spectral_radius(G)
    hH, _ = cheeger_cut(H)
    rhoH = se.spectral_radius(H)
    axes[0].set_title(f"edgecount: {G.number_of_edges()}, cheeger: {hG:.4f}, s.r.: {rhoG:.4f}")
    nx.draw(G, pos1, with_labels=True, node_color='lightblue', edge_color='black', ax=axes[0])
    axes[1].set_title(f"edgecount: {H.number_of_edges()}, cheeger: {hH:.4f}, s.r.: {rhoH:.4f}")
    nx.draw(H, pos2, with_labels=True, node_color='lightblue', edge_color='black', ax=axes[1])

    plt.suptitle(f"{nameG} -> {nameH}")
    plt.tight_layout()
    plt.show()


def visualize_partitioned_cut(g, cut_set):
    a_nodes = list(cut_set)
    b_nodes = [v for v in g.nodes() if v not in cut_set]
    a = g.subgraph(a_nodes)
    b = g.subgraph(b_nodes)
    pos = nx.spring_layout(g, seed=42)
    pos_a = nx.spring_layout(a, seed=1)
    pos_b = nx.spring_layout(b, seed=2)
    offset = np.array([2.5, 0])
    for key in pos_b:
        pos_b[key] += offset
    pos_split = {**pos_a, **pos_b}
    cut_edges = [(u, v) for u, v in g.edges() if (u in cut_set) != (v in cut_set)]
    internal_edges = [(u, v) for u, v in g.edges() if (u in cut_set) == (v in cut_set)]
    h_global, _ = cheeger_cut(g)
    rho_g = se.spectral_radius(g)
    rho_a = se.spectral_radius(a) if a.number_of_nodes() > 1 else 0
    rho_b = se.spectral_radius(b) if b.number_of_nodes() > 1 else 0
    h_a, _ = cheeger_cut(a) if a.number_of_nodes() > 1 else (0, None)
    h_b, _ = cheeger_cut(b) if b.number_of_nodes() > 1 else (0, None)
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    ax = axes[0]
    nx.draw_networkx_nodes(g, pos, nodelist=a_nodes, node_color='skyblue', ax=ax)
    nx.draw_networkx_nodes(g, pos, nodelist=b_nodes, node_color='lightgreen', ax=ax)
    nx.draw_networkx_labels(g, pos, ax=ax)
    nx.draw_networkx_edges(g, pos, edgelist=internal_edges, edge_color='gray', ax=ax)
    nx.draw_networkx_edges(g, pos, edgelist=cut_edges, edge_color='red', width=1, ax=ax)
    ax.set_title(f"global partition\nρ: {rho_g:.4f}, h: {h_global:.4f}\ncut edges: {len(cut_edges)}")
    ax.axis('off')
    ax = axes[1]
    nx.draw_networkx_nodes(g, pos_split, nodelist=a_nodes, node_color='skyblue', ax=ax)
    nx.draw_networkx_nodes(g, pos_split, nodelist=b_nodes, node_color='lightgreen', ax=ax)
    nx.draw_networkx_labels(g, pos_split, ax=ax)
    nx.draw_networkx_edges(g, pos_split, edgelist=internal_edges, edge_color='gray', ax=ax)
    nx.draw_networkx_edges(g, pos_split, edgelist=cut_edges, edge_color='orange', style='dashed', width=1, ax=ax)
    ax.set_title(f"individual partitions\nρ(a): {rho_a:.4f}, h(a): {h_a:.4f} | ρ(b): {rho_b:.4f}, h(b): {h_b:.4f}")
    ax.axis('off')
    plt.tight_layout()
    plt.show()
    return a, b

def kelmans_op(g, u, v):
    if not g.has_edge(u, v):
        return g.copy()
    h = g.copy()
    for w in list(h.neighbors(u)):
        if w != v and not h.has_edge(v, w):
            h.add_edge(v, w)
            h.remove_edge(u, w)
    return h

def visualize_kelmans_operation(g, u, v):
    h = kelmans_op(g, u, v)
    pos_orig = nx.kamada_kawai_layout(g)
    pos_after = nx.kamada_kawai_layout(h)
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    axes[0].set_title(f"original graph\nedges: {g.number_of_edges()}")
    nx.draw(g, pos_orig, with_labels=True, node_color='lightblue', edge_color='gray', ax=axes[0])
    axes[1].set_title(f"after kelmans\nedges: {h.number_of_edges()}\nspectral rad: {se.spectral_radius(h):.4f}")
    nx.draw(h, pos_after, with_labels=True, node_color='lightgreen', edge_color='gray', ax=axes[1])
    plt.tight_layout()
    plt.show()

def draw_spectral_embedding(g, method='fiedler'):
    if method == 'fiedler':
        data = se.fast_eigen_decomp(g)
        fiedler = data.get("fiedler_vector")
        if fiedler is None:
            print("fiedler vector not available")
            return
        fiedler = fiedler.flatten()
        plt.figure(figsize=(10, 5))
        plt.scatter(range(len(fiedler)), fiedler, c='blue')
        plt.xlabel("node index")
        plt.ylabel("fiedler value")
        plt.title("spectral embedding (fiedler vector)")
        plt.grid(True)
        plt.show()
    else:
        print(f"method '{method}' not implemented")

def draw_refinement_results(results):
    for name, res in results.items():
        h = res['graph']
        cut = res['cut']
        cheeger_val = res['cheeger']
        rho = res['spectral_radius']
        edge_count = res['edge_count']
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        pos_orig = nx.kamada_kawai_layout(h)
        pos_refined = nx.kamada_kawai_layout(h)
        cut_edges = [(u, v) for u, v in h.edges() if (u in cut) != (v in cut)]
        nx.draw(h, pos_orig, ax=axes[0], with_labels=True, node_color='lightblue', edge_color='gray')
        nx.draw_networkx_edges(h, pos_orig, edgelist=cut_edges, edge_color='pink', width=1, ax=axes[0])
        axes[0].set_title("refined graph (original view)")
        axes[0].axis('off')
        nx.draw(h, pos_refined, ax=axes[1], with_labels=True, node_color='lightgreen', edge_color='gray')
        nx.draw_networkx_edges(h, pos_refined, edgelist=cut_edges, edge_color='orange', width=1, ax=axes[1])
        axes[1].set_title(f"{name}\ncheeger ≈ {cheeger_val:.4f}, spectral rad ≈ {rho:.4f}, edges: {edge_count}")
        axes[1].axis('off')
        plt.tight_layout()
        plt.show()