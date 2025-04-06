import networkx as nx
import matplotlib.pyplot as plt
from collections import defaultdict
from itertools import combinations
import concurrent.futures
import cluster_refinement as refine
import numpy as np

DEBUG = False

def debug_print(*args, **kwargs):
    if DEBUG:
        print(*args, **kwargs)

def kelmans_op(G, u, v):
    debug_print(f"before op, {G.number_of_nodes()} nodes and {G.number_of_edges()} edges.")
    if not G.has_edge(u, v):
        debug_print(f"no edge between {u} and {v}. returning a copy.")
        return G.copy()
    H = G.copy()
    for w in list(H.neighbors(u)):
        if w != v and not H.has_edge(v, w):
            H.add_edge(v, w)
            H.remove_edge(u, w)
    debug_print(f"after kelmans_op: {H.number_of_nodes()} nodes and {H.number_of_edges()} edges.")
    return H

def analyze_pair(args):
    G, u, v, original_edges, original_cycles, degrees, orig_cheeger, orig_spectral = args
    # Order nodes by degree.
    a, b = (u, v) if degrees[u] > degrees[v] else (v, u)
    
    debug_print(f"analyze_pair: processing pair ({a}, {b}).")
    H = refine.kelmans_op(G, a, b)
    
    # Check that node counts remain consistent
    if H.number_of_nodes() != G.number_of_nodes():
        debug_print(f"analyze_pair: warning! node count changed from {G.number_of_nodes()} to {H.number_of_nodes()} for pair ({a}, {b}).")
    
    H_edges = set(tuple(sorted(e)) for e in H.edges())
    
    if H_edges == original_edges:
        debug_print(f"analyze_pair: Pair ({a}, {b}) - no effect.")
        return 'no_effect', (a, b), orig_cheeger, orig_spectral

    if nx.is_isomorphic(G, H):
        debug_print(f"analyze_pair: Pair ({a}, {b}) - graph is isomorphic.")
        return 'structurally_equivalent', (a, b), orig_cheeger, orig_spectral

    if not nx.is_connected(H):
        debug_print(f"pair ({a},{b}) - disconnecting")
        cat = 'disconnecting'
    elif any(d == 1 for _, d in H.degree()):
        debug_print(f"pair ({a},{b}) - pendent creating")
        cat = 'pendent_created'
    elif list(nx.bridges(H)):
        debug_print(f"pair ({a},{b}) - cut edge introduced")
        cat = 'cut_edge_introduced'
    elif len(nx.cycle_basis(H)) < original_cycles:
        debug_print(f"pair ({a},{b}) - cycle removing")
        cat = 'cycle_removed'
    else:
        cat = 'other'

    try:
        cheeger_value, _ = refine.cheeger_cut(H)
    except Exception as e:
        debug_print(f"analyze_pair: error computing cheeger for pair ({a}, {b}): {e}")
        cheeger_value = float('nan')
    try:
        spectral_value = refine.spectral_radius(H)
    except Exception as e:
        debug_print(f"analyze_pair: error computing spectral radius for pair ({a}, {b}): {e}")
        spectral_value = float('nan')
    
    return cat, (a, b), cheeger_value, spectral_value

def kelmans_sensitivity_analysis_optimized(data, max_workers=None):
    results = {
        'disconnecting': [],
        'cut_edge_introduced': [],
        'pendent_created': [],
        'no_effect': [],
        'structurally_equivalent': [],
        'cycle_removed': [],
        'other': [],
    }
    metrics = defaultdict(list)
    
    G = data['graph']
    orig_cheeger = data['cheeger']
    orig_spectral = data['spectral_radius']
    
    debug_print(f"kelmans_sensitivity_analysis: graph has {G.number_of_nodes()} nodes and {G.number_of_edges()} edges.")
    
    original_edges = set(tuple(sorted(e)) for e in G.edges())
    original_cycles = len(nx.cycle_basis(G))
    degrees = dict(G.degree())
    
    pair_args = [
        (G, u, v, original_edges, original_cycles, degrees, orig_cheeger, orig_spectral)
        for u, v in combinations(G.nodes(), 2)
    ]
    debug_print(f"kelmans_sensitivity_analysis for {G}: total pairs to analyze:", len(pair_args))
    
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(analyze_pair, args) for args in pair_args]
        for future in concurrent.futures.as_completed(futures):
            try:
                cat, pair, cheeger_val, spectral_val = future.result()
                results[cat].append(pair)
                metrics['cheeger'].append(cheeger_val)
                metrics['spectral_radius'].append(spectral_val)
            except Exception as e:
                debug_print("kelmans_sensitivity_analysis: error processing a pair:", e)
    
    return results, metrics

def compute_fragility(G, results):
    weights = {
        'pendent_created': 0.6,
        'cut_edge_introduced': 0.6,
        'cycle_removed': 0.6,
        'disconnecting': 1.0,
        'other': 0.05
    }
    fragility = {node: 0 for node in G.nodes()}
    for cat, w in weights.items():
        for u, v in results.get(cat, []):
            fragility[u] += w
            fragility[v] += w
    return fragility

def plot_graph_with_fragility(G, fragility, metrics, strategy = ""):
    max_frag = max(fragility.values()) or 1
    node_colors = {node: (0, 0, fragility[node] / max_frag) for node in G.nodes()}
    pos = nx.spring_layout(G)
    plt.figure(figsize=(8, 6))
    nx.draw(G, pos, with_labels=True,
            node_color=[node_colors[n] for n in G.nodes()], edge_color='gray')
    edge_count = G.number_of_edges()
    cheeger_vals = [v for v in metrics['cheeger'] if not (isinstance(v, float) and v != v)]
    spectral_vals = [v for v in metrics['spectral_radius'] if not (isinstance(v, float) and v != v)]
    avg_cheeger = sum(cheeger_vals) / len(cheeger_vals) if cheeger_vals else float('nan')
    avg_spectral = sum(spectral_vals) / len(spectral_vals) if spectral_vals else float('nan')
    plt.title(f"strat: {strategy}, edges: {edge_count},\n cheeger: {avg_cheeger:.3f}, spectral rad: {avg_spectral:.3f}")
    plt.show()

def analyze_graphs(res, max_workers_per_graph=None, overall_workers=None):
    all_results = {}
    all_metrics = {}
    strategies = list(res.keys())
    data_list = list(res.values())
    for i, data in enumerate(data_list):
        G = data['graph']
        debug_print(f"graph {i}: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges.")
    
    with concurrent.futures.ProcessPoolExecutor(max_workers=overall_workers) as executor:
        future_to_graph = {executor.submit(kelmans_sensitivity_analysis_optimized, data, max_workers_per_graph): i
                           for i, data in enumerate(data_list)}
        for future in concurrent.futures.as_completed(future_to_graph):
            idx = future_to_graph[future]
            try:
                results, metrics = future.result()
                all_results[strategies[idx]] = results
                all_metrics[strategies[idx]] = metrics
                G = data_list[idx]['graph']
                fragility = compute_fragility(G, results)
                strategy = strategies[idx]
                if strategy not in ("original","closed"):
                    plot_graph_with_fragility(G, fragility, metrics, strategy)
            except Exception as e:
                debug_print(f"analyze_graphs: error processing graph {idx}: {e}")
    return all_results, all_metrics

categories = ['disconnecting', 'cut_edge_introduced', 'pendent_created', 'no_effect', 'structurally_equivalent', 'cycle_removed', 'other']
fragility_weights = {'disconnecting': 1.0, 'cut_edge_introduced': 0.6, 'pendent_created': 0.6, 'no_effect': 0.0, 'structurally_equivalent': 0.0, 'cycle_removed': 0.6, 'other': 0.05}