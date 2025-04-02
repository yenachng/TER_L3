import networkx as nx
import numpy as np
from collections import deque
import random

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
            print(f"added edge ({u}, {w})")
        H.remove_node(v)
        print(f"removed node {v}")
        for x in (u,w):
            if x in H and H.degree(x) == 2 and x not in candidates:
                candidates.append(x)
    return H

def pad_pendents(G):
    H = G.copy()
    candidates = [v for v in H.nodes if H.degree(v)<2]
    if candidates==[]:
        return H
    else:
        for p in candidates:
            for u in H.nodes():
                if u !=p and u not in H.neighbors(p):
                    H.add_edge(p, u)
                    break
    return H
    