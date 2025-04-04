import networkx as nx
import numpy as np
import random
from collections import deque
from multiprocessing import Pool
from itertools import combinations

import spectral_embedding as spe
from data import test_data
H = random.choice(list(test_data.values()))

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

def edges_equal(G, H):
    return set(tuple(sorted(e)) for e in G.edges()) == set(tuple(sorted(e)) for e in H.edges())

