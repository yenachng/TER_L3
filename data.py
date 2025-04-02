import os
import gzip
import networkx as nx

def load_hcp(filepath, index_offset=0):
    G = nx.Graph()
    with gzip.open(filepath, 'rt', encoding='utf-8') as f:
        edge_section = False
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            # Look for the start of the edge section
            if not edge_section:
                if "EDGE" in line and "SECTION" in line:
                    edge_section = True
                continue
            # Process lines in the edge section
            if line in ["-1", "EOF"]:
                break
            tokens = line.split()
            if len(tokens) < 2:
                continue
            try:
                source, target = map(int, tokens[:2])
                # Apply index offset if needed
                source += index_offset
                target += index_offset
                G.add_edge(source, target)
            except ValueError:
                # Skip lines that cannot be parsed
                continue
    return G

def load_all_graphs(folder_path, index_offset=0):
    files = [f for f in os.listdir(folder_path) if f.endswith('.hcp') or f.endswith('.hcp.gz')]
    graphs = {}
    for file in files:
        filepath = os.path.join(folder_path, file)
        graphs[file] = load_hcp(filepath, index_offset=index_offset)
    return graphs

folder = "test_data"  
test_data = load_all_graphs(folder, index_offset=0)  # Set index_offset to -1 if needed
for filename, graph in test_data.items():
    print(f"Graph {filename}: {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges")
