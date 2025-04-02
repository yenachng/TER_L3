from collections import deque

def aggressive_pruning(G, threshold=2):
    H = G.copy()
    n = G.number_of_nodes()
    candidates = deque(list(H.edges()))
    while candidates:
        u, v = candidates.popleft()
        if not H.has_edge(u, v): #remove if edge doesn't exist anymore
            continue
        if H.degree(u) + H.degree(v) < n + threshold:
            continue
        else:
            H.remove_edge(u, v)
            #print(f"removing edge {u}-{v}")
            #draw(H)
            for w in list(H.neighbors(u)):
                candidates.append((u, w))
            for w in list(H.neighbors(v)):
                candidates.append((v, w))
    
    #plt.show()
    return H