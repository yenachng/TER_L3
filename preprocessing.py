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
    