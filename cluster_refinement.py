import networkx as nx
import numpy as np
import random
from collections import deque
from multiprocessing import Pool

import spectral_embedding as spe
from data import test_data
H = random.choice(list(test_data.values()))

def bondy_chvatal_closure(G):
    n = G.number_of_nodes()
    nodes = list(G.nodes())
    neighbor = {node : set(G.neighbors(node)) for node in G.nodes()}
    all_nodes = set(nodes)
    non_neighbors = {node : all_nodes - neighbor[node] - {node} for node in G.nodes()}
    queue = deque(nodes)
    in_queue = {u: True for u in nodes}
    while queue:
        u = queue.popleft()
        in_queue[u] = False
        for v in list(non_neighbors[u]):
            if len(neighbor[u] & neighbor[v]) > n:
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
    print("closure complete")
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

        transferable_uv = len(neighbors_u-neighbors_v)
        transferable_vu = len(neighbors_v-neighbors_u)
        deg_u = G.degree(u)
        deg_v = G.degree(v)
        score_uv = transferable_uv*max(deg_u-deg_v, 0)
        score_vu = transferable_vu*max(deg_v-deg_u,0)
        if score_uv > best_score:
            best_score = score_uv
            best_edge = (u, v)
        if score_vu > best_score:
            best_score = score_vu
            best_edge = (v, u)
    return best_edge

def compute_metrics(cluster):
    density  = nx.density(cluster)
    avg_clust = np.mean(list(nx.clustering(cluster).values()))
    triangles = sum(nx.triangles(cluster).values())/3
    return density, avg_clust, triangles

def refine_cluster(cluster, low_dens_threshold = 0.3):
    density, avg_clust, triangles = compute_metrics(cluster)
    degrees = dict(cluster.degree())
    min_deg = min(degrees.values()) if degrees else 0
    n = cluster.number_of_nodes()

    print(f"cluster metrics:\n density: {density:.3f}\n avg clustering: {avg_clust:.3f}\n triangles: {triangles:.0f}\n min degree: {min_deg}")

    if n>=3 and min_deg >= n/2:
        print("cluster meets dirac's condition for hamiltonicity, no refinement needed.")
        return cluster
    
    if density < low_dens_threshold:
        print("cluster is sparse. applying bondy-chvatal for closure.")
        return bondy_chvatal_closure(cluster)
    
    else:
        print("cluster is moderately connected. applying kelman's operation.")
