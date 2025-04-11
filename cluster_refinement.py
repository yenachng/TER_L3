import networkx as nx
import numpy as np
import random
import spectral_embedding as se
import scipy.sparse as sp
import heapq as hq
from collections import deque
from itertools import combinations
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

def bondy_chvatal_closure(g):
    n = g.number_of_nodes()
    nodes = list(g.nodes())
    nb = {node: set(g.neighbors(node)) for node in nodes}
    all_nodes = set(nodes)
    non_nb = {node: all_nodes - nb[node] - {node} for node in nodes}
    q = deque(nodes)
    in_q = {node: True for node in nodes}
    while q:
        u = q.popleft()
        in_q[u] = False
        for v in list(non_nb[u]):
            if len(nb[u]) + len(nb[v]) >= n:
                nb[u].add(v)
                nb[v].add(u)
                non_nb[u].remove(v)
                non_nb[v].remove(u)
                if not in_q[u]:
                    q.append(u)
                    in_q[u] = True
                if not in_q[v]:
                    q.append(v)
                    in_q[v] = True
    h = nx.Graph()
    for u in nb:
        for v in nb[u]:
            if u < v:
                h.add_edge(u, v)
    return h

def cheeger_cut(g):
    if g.number_of_nodes() == 0:
        return float('inf'), set()
    a = se.to_adjacency(g)
    degrees = np.array(a.sum(axis=1)).flatten()
    total_vol = degrees.sum()
    l = np.diag(degrees) - a
    _, vecs = sp.linalg.eigsh(l, k=2, which='SM')
    fiedler = vecs[:, 1]
    node_list = list(g.nodes())
    order = np.argsort(fiedler)
    sorted_nodes = [node_list[i] for i in order]
    cum_vol = np.cumsum(degrees[order])
    best_ratio = float('inf')
    best_idx = None
    for i in range(1, len(sorted_nodes)):
        s = set(sorted_nodes[:i])
        boundary = sum(1 for u, v in g.edges() if (u in s) ^ (v in s))
        vol_s = cum_vol[i - 1]
        vol_not = total_vol - vol_s
        if vol_s > 0 and vol_not > 0:
            ratio = boundary / min(vol_s, vol_not)
            if ratio < best_ratio:
                best_ratio = ratio
                best_idx = i
    best_cut = set(sorted_nodes[:best_idx]) if best_idx is not None else set(node_list)
    return best_ratio, best_cut

def boundary_edges(g, cut_set):
    cut_set = set(cut_set)
    return [(u, v) for u, v in g.edges() if (u in cut_set) != (v in cut_set)]

def get_edge_attributes(g, cut_set):
    deg = dict(g.degree())
    boundary = {(min(u, v), max(u, v)): (u in cut_set) != (v in cut_set) for u, v in g.edges()}
    deg_diff = {(min(u, v), max(u, v)): abs(deg[u] - deg[v]) for u, v in g.edges()}
    return boundary, deg_diff

def combined_edge_priority(e, features, weights, normalize=False):
    key = (min(e[0], e[1]), max(e[0], e[1]))
    b, d = features['boundary'][key], features['deg_diff'][key]
    if normalize:
        d /= max(features['deg_diff'].values()) or 1
    return weights['boundary'] * int(b) + weights['deg_diff'] * d

def aggressive_pruning_ordered(g, weights, normalize=False):
    h = g.copy()
    n = g.number_of_nodes()
    _, cut = cheeger_cut(g)
    features = {}
    features['boundary'], features['deg_diff'] = get_edge_attributes(g, cut)
    seen, heap = set(), []
    for e in h.edges():
        prio = combined_edge_priority(e, features, weights, normalize)
        hq.heappush(heap, (prio, e))
    while heap:
        _, (u, v) = hq.heappop(heap)
        eid = (min(u, v), max(u, v))
        if eid in seen or not h.has_edge(u, v):
            continue
        seen.add(eid)
        if h.degree(u) + h.degree(v) < n + 2:
            continue
        h.remove_edge(u, v)
        for w in list(h.neighbors(u)) + list(h.neighbors(v)):
            ew = (min(w, u), max(w, u)) if w != u else (min(w, v), max(w, v))
            if ew not in seen and h.has_edge(*ew):
                prio = combined_edge_priority(ew, features, weights, normalize)
                hq.heappush(heap, (prio, ew))
    return h

def greedy_cycle_pathfinder(g, spectral_info=None):
    nodes = list(g.nodes())
    if not nodes:
        return []
    start = nodes[0]
    path = [start]
    visited = {start}
    current = start
    while True:
        nbrs = list(g.neighbors(current))
        if spectral_info is not None and current in spectral_info:
            current_val = spectral_info[current]
            nbrs.sort(key=lambda v: abs(spectral_info.get(v, 0) - current_val))
        found = False
        for nbr in nbrs:
            if nbr not in visited:
                path.append(nbr)
                visited.add(nbr)
                current = nbr
                found = True
                break
        if not found:
            break
    if g.has_edge(path[-1], start):
        path.append(start)
        return path
    return path

def greedy_partial_cycle_extractor(g, spectral_info=None, min_cycle_length=4):
    cycles = []
    h = g.copy()
    while h.number_of_nodes() > 2:
        cycle = greedy_cycle_pathfinder(h, spectral_info)
        if cycle is None or len(cycle) < min_cycle_length:
            break
        cycles.append(cycle)
        rep = cycle[0]
        h.remove_nodes_from(set(cycle) - {rep})
    return cycles, h

def find_hamiltonian_cycle_in_cluster_graph(h):
    nodes = list(h.nodes())
    n = len(nodes)
    if n == 0:
        return []
    if n == 1:
        return [nodes[0], nodes[0]]
    visited = {node: False for node in nodes}
    path = []
    def dfs(u, depth):
        path.append(u)
        visited[u] = True
        if depth == n:
            if h.has_edge(u, path[0]):
                path.append(path[0])
                return True
            visited[u] = False
            path.pop()
            return False
        for w in h.neighbors(u):
            if not visited[w]:
                if dfs(w, depth + 1):
                    return True
        visited[u] = False
        path.pop()
        return False
    if dfs(nodes[0], 1):
        return path
    return None


def compute_cluster_frontier(cluster_nodes, clusters_dict, g):
    front = {}
    cl_set = set(cluster_nodes)
    for v in cluster_nodes:
        ext = set(g.neighbors(v)) - cl_set
        if not ext:
            continue
        if clusters_dict:
            ext_lbl = set()
            for u in ext:
                for cid, nodes in clusters_dict.items():
                    if u in nodes:
                        ext_lbl.add(cid)
            if ext_lbl:
                front[v] = ext_lbl
        else:
            front[v] = ext
    return front

def recursive_cheeger_split(g, frontier=None, min_size=5, cond_threshold=0.4):
    if frontier is None:
        frontier = []
    n = g.number_of_nodes()
    if n == 0 or n <= min_size:
        return [(set(g.nodes()), frontier, g)]
    cond, cut_set = cheeger_cut(g)
    boundary = boundary_edges(g, cut_set)
    frontier.append(boundary)
    if cond is None or cond > cond_threshold or cut_set == set(g.nodes()) or not cut_set:
        return [(set(g.nodes()), frontier, g)]
    comp_left = g.subgraph(cut_set).copy()
    comp_right = g.subgraph(set(g.nodes()) - cut_set).copy()
    clusters_left = recursive_cheeger_split(comp_left, frontier, min_size, cond_threshold)
    clusters_right = recursive_cheeger_split(comp_right, frontier, min_size, cond_threshold)
    return clusters_left + clusters_right

def build_supernode_graph_from_labelmap(g, node_label_map):
    clusters = {}
    for node, lbl in node_label_map.items():
        clusters.setdefault(lbl, set()).add(node)
    def define_frontier(cluster, clusters, g):
        node_to_cluster = {node: cid for cid, nodes in clusters.items() for node in nodes}
        frontier = {}
        for v in cluster:
            ext_clusters = {node_to_cluster[u] for u in g.neighbors(v)
                            if u in node_to_cluster and node_to_cluster[u] != node_to_cluster[v]}
            if len(ext_clusters) >= 2:
                frontier[v] = ext_clusters
        return frontier
    h = nx.Graph()
    for cid, nodes in clusters.items():
        front = define_frontier(nodes, clusters, g)
        h.add_node(cid, cluster=nodes, frontier=front)
    for cid1 in clusters:
        for cid2 in clusters:
            if cid1 < cid2:
                connected = False
                front1 = h.nodes[cid1]['frontier']
                for v in front1:
                    for u in g.neighbors(v):
                        if u in clusters[cid2]:
                            connected = True
                            break
                    if connected:
                        break
                if connected:
                    h.add_edge(cid1, cid2)
    return h

def labels_to_labelmap(g, labels):
    return {node: lbl for node, lbl in zip(g.nodes(), labels)}
def labels_to_clusters(g, labels):
    clusters = {}
    for node, label in zip(g.nodes(), labels):
        clusters.setdefault(label, set()).add(node)
    return clusters

def variable_orderings(g, visualisation=True):
    g = bondy_chvatal_closure(g)
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
            edges = list(g.edges())
            random.shuffle(edges)
            h = g.copy()
            for u, v in edges:
                if h.degree(u) + h.degree(v) >= g.number_of_nodes() + 2:
                    h.remove_edge(u, v)
        elif name == 'fiedler':
            lmat = nx.normalized_laplacian_matrix(g).astype(float)
            _, vecs = sp.linalg.eigsh(lmat, k=2, which='SM')
            fiedler = np.real(vecs[:, 1])
            node_idx = {node: i for i, node in enumerate(g.nodes())}
            fiedler_diff = {
                (min(u, v), max(u, v)): abs(fiedler[node_idx[u]] - fiedler[node_idx[v]])
                for u, v in g.edges()
            }
            weights = fiedler_diff
            h = g.copy()
            for u, v in sorted(g.edges(), key=lambda e: weights.get((min(e[0], e[1]), max(e[0], e[1])), 0), reverse=True):
                if h.degree(u) + h.degree(v) >= g.number_of_nodes() + 2:
                    h.remove_edge(u, v)
        else:
            _, cut = cheeger_cut(g)
            bd, dd = get_edge_attributes(g, cut)
            weights = strat['weights']
            h = aggressive_pruning_ordered(g, weights, normalize=True)
        h_end, cut_end = cheeger_cut(h)
        rho = se.spectral_radius(h)
        results[name] = {
            'cheeger': h_end,
            'spectral_radius': rho,
            'edge_count': h.number_of_edges(),
            'graph': h,
            'cut': cut_end
        }
        if visualisation:
            fig, axes = plt.subplots(1, 2, figsize=(12, 5))
            pos = nx.kamada_kawai_layout(g)
            pos_h = nx.kamada_kawai_layout(h)
            cut_edges_g = [(u, v) for u, v in g.edges() if (u in cut) != (v in cut)]
            cut_edges_h = [(u, v) for u, v in h.edges() if (u in cut) != (v in cut)]
            nx.draw(g, pos, ax=axes[0], with_labels=True, node_color='lightblue', edge_color='gray')
            nx.draw_networkx_edges(g, pos, edgelist=cut_edges_g, edge_color='pink', width=1, ax=axes[0])
            axes[0].set_title(f"original\nedges = {g.number_of_edges()}")
            axes[0].axis('off')
            nx.draw(h, pos_h, ax=axes[1], with_labels=True, node_color='lightgreen', edge_color='gray')
            nx.draw_networkx_edges(h, pos_h, edgelist=cut_edges_h, edge_color='orange', width=1, ax=axes[1])
            axes[1].set_title(f"{name}\ncheeger ≈ {h_end:.4f}, ρ ≈ {rho:.4f}, edges = {h.number_of_edges()}")
            axes[1].axis('off')
            plt.tight_layout()
            plt.show()
    return results


def select_kelmans_candidate(g):
    best_score, best_edge = -1, None
    deg = dict(g.degree())
    for u, v in g.edges():
        nu, nv = set(g.neighbors(u)), set(g.neighbors(v))
        score_uv = len(nu - nv) * max(g.degree(u) - g.degree(v), 0) / g.degree(u)
        score_vu = len(nv - nu) * max(g.degree(v) - g.degree(u), 0) / g.degree(v)
        if score_uv > best_score:
            best_score, best_edge = score_uv, (u, v)
        if score_vu > best_score:
            best_score, best_edge = score_vu, (v, u)
    return best_edge

def all_kelmans_res(g):
    groups = {
        "no_effect": [], "structurally_equivalent": [], "disconnecting": [],
        "cut_edge_introduced": [], "pendent_created": [], "cycle_removed": [], "other": []
    }
    orig_edges = {tuple(sorted(e)) for e in g.edges()}
    orig_cycles = len(nx.cycle_basis(g))
    degs = dict(g.degree())
    for u, v in combinations(g.nodes(), 2):
        a, b = (u, v) if degs[u] > degs[v] else (v, u)
        h = kelmans_op(g, a, b)
        h_edges = {tuple(sorted(e)) for e in h.edges()}
        if h_edges == orig_edges:
            groups["no_effect"].append((a, b))
            continue
        if nx.is_isomorphic(g, h):
            groups["structurally_equivalent"].append((a, b))
            continue
        if not nx.is_connected(h):
            groups["disconnecting"].append((a, b))
            continue
        if any(deg == 1 for _, deg in h.degree()):
            groups["pendent_created"].append((a, b))
            continue
        if list(nx.bridges(h)):
            groups["cut_edge_introduced"].append((a, b))
            continue
        if len(nx.cycle_basis(h)) < orig_cycles:
            groups["cycle_removed"].append((a, b))
            continue
        groups["other"].append((a, b))
    return groups