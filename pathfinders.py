import networkx as nx
import numpy as np
import random
from example_graphs import L_k_n, N_k_n

def frontier_mapping(n, k, s):
    w = nx.complete_graph(n - k)
    h = nx.Graph()
    h_nodes = list(range(n - k, n))
    h.add_nodes_from(h_nodes)
    g = nx.compose(w, h)
    if s == 1:
        return L_k_n(k, n)
    elif s == k:
        return N_k_n(k, n)
    else:
        vertex_colors = {}
        edge_colors = {}
        d_h = k - s
        f_vertices = sorted(w.nodes())[:s]
        for v in f_vertices:
            for h_node in sorted(h.nodes()):
                g.add_edge(v, h_node)
        h_nodes_sorted = sorted(h.nodes())
        for i in range(min(k, len(h_nodes_sorted))):
            h_node = h_nodes_sorted[i]
            other_h = [x for x in h_nodes_sorted if x != h_node]
            chosen = other_h if d_h >= len(other_h) else random.sample(other_h, d_h)
            for neighbor in chosen:
                h.add_edge(h_node, neighbor)
                g.add_edge(h_node, neighbor)
        for v in g.nodes():
            if v < (n - k):
                vertex_colors[v] = 'red' if v < s else 'gray'
            else:
                vertex_colors[v] = 'green'
        h_edges = {tuple(sorted(edge)) for edge in h.edges()}
        w_edges = {tuple(sorted(edge)) for edge in w.edges()}
        for edge in g.edges():
            e_sorted = tuple(sorted(edge))
            if e_sorted in h_edges:
                edge_colors[edge] = 'green'
            elif e_sorted in w_edges:
                edge_colors[edge] = 'black'
            else:
                edge_colors[edge] = 'red'
        return g, vertex_colors, edge_colors

def path_find(current, path, visited, g):
    for neighbor in g.neighbors(current):
        if neighbor not in visited:
            path.append(neighbor)
            visited.add(neighbor)
            return path_find(neighbor, path, visited, g)
    return path

def a_path(g):
    nodes = list(g.nodes())
    if not nodes:
        return []
    start = nodes[0]
    path = [start]
    visited = {start}
    return path_find(start, path, visited, g)

def rotation(path, i):
    return path[:i+1] + list(reversed(path[i+1:]))

def path_finder(path, g, n):
    if set(path) == set(g.nodes()):
        return path
    else:
        visited = set(path)
        prev = path[-2] if len(path) >= 2 else None
        neighbors = set(g.neighbors(path[-1]))
        access = neighbors - {prev} if prev is not None else neighbors
        access = access.intersection(visited)
        if not access:
            raise ValueError("no accessible vertices")
        unvisited = set(g.nodes()) - visited
        end_access = []
        for v in access:
            idx = path.index(v)
            if idx < len(path) - 1:
                next_v = path[idx+1]
                count = len(set(g.neighbors(next_v)) - visited)
            else:
                count = len(set(g.neighbors(v)) - visited)
            end_access.append((v, count))
        end_access.sort(key=lambda tup: tup[1], reverse=True)
        v = end_access[0][0]
        i = path.index(v)
        rot_path = rotation(path, i)
        tail = path_find(rot_path[-1], list(rot_path), set(rot_path), g)
        if len(tail) <= len(path):
            raise ValueError("no progress :(")
        return path_finder(tail, g, n)

def dfs_pathfinder(g, path=None, visited=None):
    n = g.number_of_nodes()
    if path is None:
        start = max(g.nodes(), key=lambda v: g.degree(v))
        path = [start]
        visited = {start}
    if len(path) == n:
        return path
    current = path[-1]
    for neighbor in g.neighbors(current):
        if neighbor not in visited:
            path.append(neighbor)
            visited.add(neighbor)
            res = dfs_pathfinder(g, path, visited)
            if res is not None and len(res) == n:
                return res
            path.pop()
            visited.remove(neighbor)
    return None

def remove_edge(v, g, k):
    if g.degree(v) <= k:
        return v
    else:
        neighbor = random.choice(list(g.neighbors(v)))
        g.remove_edge(v, neighbor)
        remove_edge(v, g, k)

def kconnected(n, k):
    g = nx.complete_graph(n)
    for v in list(g.nodes()):
        m = random.randint(k, n - 1)
        remove_edge(v, g, m)
    return g

def order_neighbors_with_spectral(g, node, spectral_info):
    nbrs = list(g.neighbors(node))
    if node not in spectral_info:
        random.shuffle(nbrs)
        return nbrs
    current_val = spectral_info[node]
    nbrs.sort(key=lambda v: abs(spectral_info.get(v, 0) - current_val))
    return nbrs

def greedy_path_extend(current, path, visited, g, spectral_info):
    nbrs = order_neighbors_with_spectral(g, current, spectral_info)
    for nbr in nbrs:
        if nbr not in visited:
            path.append(nbr)
            visited.add(nbr)
            return greedy_path_extend(nbr, path, visited, g, spectral_info)
    return path

def greedy_rotate_cycle(path, g, spectral_info):
    if set(path) == set(g.nodes()):
        if g.has_edge(path[-1], path[0]):
            return path
    current = path[-1]
    for i in range(len(path) - 2, -1, -1):
        if g.has_edge(current, path[i]):
            new_path = rotation(path, i)
            new_extended = greedy_path_extend(new_path[-1], new_path.copy(), set(new_path), g, spectral_info)
            return new_extended
    return path

def greedy_cycle_pathfinder(g, spectral_info):
    nodes = list(g.nodes())
    if not nodes:
        return []
    start = nodes[0]
    path = [start]
    visited = {start}
    path = greedy_path_extend(start, path, visited, g, spectral_info)
    previous_length = len(path)
    while True:
        new_path = greedy_rotate_cycle(path, g, spectral_info)
        if len(new_path) <= previous_length:
            break
        previous_length = len(new_path)
        path = new_path
    if g.has_edge(path[-1], path[0]):
        path.append(path[0])
        return path
    return path

def greedy_partial_cycle_extractor(g, spectral_info=None, min_cycle_length=4):
    cycles = []
    n = g.number_of_nodes()
    h = g.copy()
    i = 1
    j = 0
    while h.number_of_nodes() > np.log(n) and j<4:
        cycle = greedy_cycle_pathfinder(h, spectral_info)
        if cycle is None or len(cycle) < min_cycle_length:
            h.remove_nodes_from(cycle)
            i += 1
            j += 1
            continue
        cycles.append(cycle)
        h.remove_nodes_from(cycle)
        i += 1
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

def extract_cycle_path(h):
    nodes = list(h.nodes())
    if not nodes:
        return []
    start = nodes[0]
    cycle = [start]
    current = start
    prev = None
    while True:
        neighbors = list(h.neighbors(current))
        next_node = None
        for v in neighbors:
            if v != prev:
                next_node = v
                break
        if next_node is None:
            break
        if next_node == start:
            cycle.append(start)
            break
        cycle.append(next_node)
        prev, current = current, next_node
    return cycle