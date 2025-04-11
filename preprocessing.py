import networkx as nx
import numpy as np
from collections import deque

def contract_deg2(G):
    H = G.copy()
    candidates = deque([v for v in H.nodes if H.degree(v) == 2])
    while candidates:
        v = candidates.popleft()
        if v not in H or H.degree(v) != 2:
            continue
        neighbors = list(H.neighbors(v))
        if len(neighbors) != 2:
            continue
        u, w = neighbors[0], neighbors[1]
        if not H.has_edge(u, w):
            H.add_edge(u, w)
        H.remove_node(v)
        for x in (u,w):
            if x in H and H.degree(x) == 2 and x not in candidates:
                candidates.append(x)
    return H

def pad_pendents(G):
    H = G.copy()
    valid_nodes = [v for v in H.nodes() if isinstance(v, (int, str, tuple))]
    pendents = deque([v for v in H.nodes() if H.degree(v) < 2])
    while pendents:
        p = pendents.popleft()
        if H.degree(p) >= 2:
            continue
        for u in valid_nodes:
            if u != p and u not in H.neighbors(p):
                H.add_edge(p, u)
                break
        pendents = deque([v for v in H.nodes() if H.degree(v) < 2])
    return H
    
def find_maximal_clique(G):
    cliques = list(nx.find_cliques(G))
    if not cliques:
        return set()
    max_clique = max(cliques, key=len)
    return set(max_clique)

def define_frontier(clique, G):
    frontier = {}
    for v in clique:
        external_neighbors = {u for u in G.neighbors(v) if u not in clique}
        if external_neighbors and len(external_neighbors) >= 2:
            frontier[v] = external_neighbors
    return frontier

def extract_dense_region(G, min_size=3):
    clique = find_maximal_clique(G)
    if len(clique) < min_size:
        return set(), {}
    frontier = define_frontier(clique, G)
    return clique, frontier

def contract_clique_to_supernode(G, clique, frontier, supernode_label):
    external_neighbors = set()
    for v in clique:
        external_neighbors.update({u for u in G.neighbors(v) if u not in clique})
    H = G.copy()
    H.remove_nodes_from(clique)
    H.add_node(supernode_label, type="supernode", original_clique=clique, frontier=frontier)
    for u in external_neighbors:
        H.add_edge(supernode_label, u)
    return H

def chain_contraction_preserving(G, attachments=None):
    if attachments is None:
        attachments = {node: set(G.neighbors(node)) for node in G.nodes()}
    H = G.copy()
    att = {node: set(neigh) for node, neigh in attachments.items()}
    changed = True
    while changed:
        changed = False
        for v in list(H.nodes()):
            if H.degree(v) == 2:
                neighbors = list(H.neighbors(v))
                if len(neighbors) == 2 and neighbors[0] != neighbors[1]:
                    u = neighbors[0]
                    new_att = att.get(u, set()).union(att.get(v, set()))
                    new_att.discard(u)
                    new_att.discard(v)
                    att[u] = new_att
                    H = nx.contracted_nodes(H, u, v, self_loops=False)
                    for x in list(att.keys()):
                        if v in att[x]:
                            att[x].remove(v)
                            att[x].add(u)
                    changed = True
                    break
    return H