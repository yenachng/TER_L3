import networkx as nx
import numpy as np
import random
import spectral_embedding as se
import scipy.sparse as sp
import heapq as hq
import matplotlib.pyplot as plt
from collections import deque
from itertools import combinations

import spectral_embedding as spe

def spectral_radius(G):
    A = se.computeA(G)
    val, _ = sp.linalg.eigsh(A, k=1, which='LA')
    return val[0]

def bondy_chvatal_closure(G):
    n = G.number_of_nodes()
    nodes = list(G.nodes())
    neighbor = {node: set(G.neighbors(node)) for node in nodes}
    all_nodes = set(nodes)
    non_neighbors = {node: all_nodes - neighbor[node] - {node} for node in nodes}
    queue = deque(nodes)
    in_queue = {node: True for node in nodes}
    while queue:
        u = queue.popleft()
        in_queue[u] = False
        for v in list(non_neighbors[u]):
            if len(neighbor[u]) + len(neighbor[v]) >= n:
                neighbor[u].add(v)
                neighbor[v].add(u)
                non_neighbors[u].remove(v)
                non_neighbors[v].remove(u)
                if not in_queue[u]:
                    queue.append(u)
                    in_queue[u] = True
                if not in_queue[v]:
                    queue.append(v)
                    in_queue[v] = True
    H = nx.Graph()
    for u in neighbor:
        for v in neighbor[u]:
            if u < v:
                H.add_edge(u, v)
    return H

def kelmans_op(G, u, v):
    if not G.has_edge(u,v):
        return G.copy()
    H = G.copy()
    for w in list(H.neighbors(u)):
        if w!=v and not H.has_edge(v,w):
            H.add_edge(v,w)
            H.remove_edge(u,w)
    return H

def select_kelmans_candidate(G):
    best_score = -1
    best_edge = None
    for u,v in G.edges():
        neighbors_u = set(G.neighbors(u))
        neighbors_v = set(G.neighbors(v))
        # kelmans u->v: les voisins de u pas attaches a v se detache de u pr s'attacher a v
        # donc la 'magnitude' de changement par u->v est les voisins de u qui ne sont pas dans le voisinage de v: N(u) - N(v)
        # d'un autre cote, 
        transferable_uv = len(neighbors_u-neighbors_v)
        transferable_vu = len(neighbors_v-neighbors_u)
        deg_u = G.degree(u)
        deg_v = G.degree(v)
        score_uv = transferable_uv*max(deg_u-deg_v, 0)/deg_u
        score_vu = transferable_vu*max(deg_v-deg_u,0)/deg_v
        if score_uv > best_score:
            best_score = score_uv
            best_edge = (u, v)
        if score_vu > best_score:
            best_score = score_vu
            best_edge = (v, u)
    return best_edge


def all_kelmans_res(G):
    groups = {
        "no_effect": [], "structurally_equivalent": [], "disconnecting": [],
        "cut_edge_introduced": [], "pendent_created": [], "cycle_removed": [],
        "other": []
    }

    original_edges = set(tuple(sorted(e)) for e in G.edges())
    original_cycles = len(nx.cycle_basis(G))
    G_degrees = dict(G.degree())

    for u, v in combinations(G.nodes(), 2):
        a, b = (u, v) if G_degrees[u] > G_degrees[v] else (v, u)
        H = kelmans_op(G, a, b)

        H_edges = set(tuple(sorted(e)) for e in H.edges())
        if H_edges == original_edges:
            groups["no_effect"].append((a, b))
            continue

        if nx.is_isomorphic(G, H):
            groups["structurally_equivalent"].append((a, b))
            continue

        if not nx.is_connected(H):
            groups["disconnecting"].append((a, b))
            continue

        if any(deg == 1 for _, deg in H.degree()):
            groups["pendent_created"].append((a, b))
            continue

        if list(nx.bridges(H)):
            groups["cut_edge_introduced"].append((a, b))
            continue

        if len(nx.cycle_basis(H)) < original_cycles:
            groups["cycle_removed"].append((a, b))
            continue

        groups["other"].append((a, b))

    return groups

def cheeger_cut(G):
    A = se.computeA(G)
    D = np.diag(A.sum(axis=1))
    L = D - A
    _, vecs = sp.linalg.eigsh(L, k=2, which='SM')
    fiedler = vecs[:, 1]

    idx = np.argsort(fiedler)
    nodes = np.array(G.nodes())[idx]
    
    in_S = np.zeros(len(nodes), dtype=bool)
    vol_S = 0
    boundary = 0
    degrees = np.array(A.sum(axis=1)).flatten()
    
    min_ratio = np.inf
    best_cut = set()
    
    for i in range(len(nodes) // 2):
        v = nodes[i]
        v_idx = list(G.nodes).index(v)
        in_S[v_idx] = True
        vol_S += degrees[v_idx]

        for u in G.neighbors(v):
            u_idx = list(G.nodes).index(u)
            if in_S[u_idx]:
                boundary -= 1
            else:
                boundary += 1

        if vol_S > 0:
            ratio = boundary / vol_S
            if ratio < min_ratio:
                min_ratio = ratio
                best_cut = set(nodes[:i+1])

    return min_ratio, best_cut

def boundary_edges(G, cut_set):
    cut_set = set(cut_set)
    return [
        (u, v)
        for u, v in G.edges()
        if (u in cut_set) != (v in cut_set)
    ]

def get_edge_attributes(G, cut_set):
    deg = dict(G.degree())
    boundary = {(min(u, v), max(u, v)): (u in cut_set) != (v in cut_set) for u, v in G.edges()}
    deg_diff = {(min(u, v), max(u, v)): abs(deg[u] - deg[v]) for u, v in G.edges()}
    return boundary, deg_diff

def combined_edge_priority(e, features, weights, normalize=False):
    if weights is None:
        return random.random()
    key = (min(e[0], e[1]), max(e[0], e[1]))
    b, d = features['boundary'][key], features['deg_diff'][key]
    if normalize:
        d /= max(features['deg_diff'].values()) or 1
    return weights['boundary'] * int(b) + weights['deg_diff'] * d

def aggressive_pruning_ordered(G, weights, normalize=False):
    H = G.copy()
    n = H.number_of_nodes()
    h, cut = cheeger_cut(G)
    features = {}
    features['boundary'], features['deg_diff'] = get_edge_attributes(G, cut)
    seen, heap = set(), []
    edge_order = []

    # Initialize the heap with all edges and their priorities.
    for e in H.edges():
        prio = combined_edge_priority(e, features, weights, normalize)
        hq.heappush(heap, (prio, e))

    while heap:
        _, (u, v) = hq.heappop(heap)
        eid = (min(u, v), max(u, v))
        if eid in seen or not H.has_edge(u, v):
            continue
        seen.add(eid)
        if H.degree(u) + H.degree(v) < n + 2:
            continue
        H.remove_edge(u, v)
        edge_order.append(eid)
        
        # Update priorities for edges incident to u.
        for w in H.neighbors(u):
            ew = (min(u, w), max(u, w))
            if ew not in seen and H.has_edge(*ew):
                prio = combined_edge_priority(ew, features, weights, normalize)
                hq.heappush(heap, (prio, ew))
        
        # Update priorities for edges incident to v.
        for w in H.neighbors(v):
            ew = (min(v, w), max(v, w))
            if ew not in seen and H.has_edge(*ew):
                prio = combined_edge_priority(ew, features, weights, normalize)
                hq.heappush(heap, (prio, ew))

    return H, edge_order

def variable_orderings(G, visualisation=True):
    strategies = [
        {'name': '(1.0, 0.0)', 'weights': {'boundary': 1.0, 'deg_diff': 0.0}},
        {'name': '(-1.0, 0.0)', 'weights': {'boundary': -1.0, 'deg_diff': 0.0}},
        {'name': '(0.0, 1.0)', 'weights': {'boundary': 0.0, 'deg_diff': 1.0}},
        {'name': '(0.0, -1.0)', 'weights': {'boundary': 0.0, 'deg_diff': -1.0}},
        {'name': '(1.0, 1.0)', 'weights': {'boundary': 1.0, 'deg_diff': 1.0}},
        {'name': '(-1.0, -1.0)', 'weights': {'boundary': -1.0, 'deg_diff': -1.0}},
        {'name': 'random', 'weights': None},
        {'name': 'fiedler', 'weights': None},
    ]

    results = {}
    for strat in strategies:
        name = strat['name']
        if name == 'random':
            edges = list(G.edges())
            random.shuffle(edges)
            H = G.copy()
            for u, v in edges:
                if H.degree(u) + H.degree(v) >= G.number_of_nodes() + 2:
                    H.remove_edge(u, v)
        elif name == 'fiedler':
            L = nx.normalized_laplacian_matrix(G).astype(float)
            _, vecs = sp.linalg.eigsh(L, k=2, which='SM')
            fiedler = np.real(vecs[:, 1])
            node_idx = {node: i for i, node in enumerate(G.nodes())}
            fiedler_diff = {
                (min(u, v), max(u, v)): abs(fiedler[node_idx[u]] - fiedler[node_idx[v]])
                for u, v in G.edges()
            }
            weights = fiedler_diff
            H = G.copy()
            for u, v in sorted(G.edges(), key=lambda e: weights.get((min(e), max(e)), 0), reverse=True):
                if H.degree(u) + H.degree(v) >= G.number_of_nodes() + 2:
                    H.remove_edge(u, v)
        else:
            h, cut = cheeger_cut(G)
            features = get_edge_attributes(G, cut)
            weights = strat['weights']
            H = aggressive_pruning_ordered(G, weights, normalize=True)

        h_end, cut_end = cheeger_cut(H)
        rho = spectral_radius(H)
        edge_count = H.number_of_edges()

        results[name] = {
            'cheeger': h_end,
            'spectral_radius': rho,
            'edge_count': edge_count,
            'graph': H,
            'cut': cut_end
        }
    if visualisation:
        for name, res in results.items():
            H = res['graph']
            cut = res['cut']
            h = res['cheeger']
            rho = res['spectral_radius']
            edge_count = res['edge_count']

            fig, axes = plt.subplots(1, 2, figsize=(12, 5))
            pos = nx.kamada_kawai_layout(G)
            pos_H = nx.kamada_kawai_layout(H)
            cut_edges_G = [(u, v) for u, v in G.edges() if (u in cut) != (v in cut)]
            cut_edges_H = [(u, v) for u, v in H.edges() if (u in cut) != (v in cut)]

            nx.draw(G, pos, ax=axes[0], with_labels=True, node_color='lightblue', edge_color='gray')
            nx.draw_networkx_edges(G, pos, edgelist=cut_edges_G, edge_color='pink', style= '-',width=1, ax=axes[0])
            axes[0].set_title("original\n"
                            f"edges = {len(edges)}")
            axes[0].axis('off')

            nx.draw(H, pos_H, ax=axes[1], with_labels=True, node_color='lightgreen', edge_color='gray')
            nx.draw_networkx_edges(H, pos_H, edgelist=cut_edges_H, edge_color='orange', width=1, ax=axes[1])
            axes[1].set_title(f"{name}\ncheeger ≈ {h:.4f}, ρ ≈ {rho:.4f}, edges = {edge_count}")
            axes[1].axis('off')

            plt.tight_layout()
            plt.show()
    return results