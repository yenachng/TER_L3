import matplotlib.pyplot as plt
import numpy as np
import matplotlib.cm as cm
import concurrent.futures
import scipy.sparse.linalg as spla
import example_graphs as examples
import cluster_refinement as refine
import networkx as nx

def generate_extremal_instances():
    instances = {}  # key: graph type, value: list of dicts {graph, params}
    instances["Petersen"] = [{"graph": nx.petersen_graph(), "params": {}}]
    # For N_k_n: valid when n >= 2*k; try k=2,3,4 and n from 2*k to 20
    nkn_list = []
    for k in range(2, 5):
        for n in range(2 * k, 21):
            try:
                G = examples.N_k_n(k, n)
                nkn_list.append({"graph": G, "params": {"k": k, "n": n}})
            except Exception as e:
                print(f"error generating N_k_n for k={k}, n={n}: {e}")
    instances["N_k_n"] = nkn_list

    # For L_k_n: valid when n >= k + 2
    lkn_list = []
    for k in range(2, 5):
        for n in range(k + 2, 21):
            try:
                G = examples.L_k_n(k, n)
                lkn_list.append({"graph": G, "params": {"k": k, "n": n}})
            except Exception as e:
                print(f"error generating L_k_n for k={k}, n={n}: {e}")
    instances["L_k_n"] = lkn_list

    # For L_k_n_bar: similar restrictions as L_k_n
    lkn_bar_list = []
    for k in range(2, 5):
        for n in range(k + 2, 21):
            try:
                G = examples.L_k_n_bar(k, n)
                lkn_bar_list.append({"graph": G, "params": {"k": k, "n": n}})
            except Exception as e:
                print(f"Error generating L_k_n_bar for k={k}, n={n}: {e}")
    instances["L_k_n_bar"] = lkn_bar_list

    # For N_k_n_bar: valid when n >= 2*k+1
    nkn_bar_list = []
    for k in range(2, 5):
        for n in range(2 * k + 1, 21):
            try:
                G = examples.N_k_n_bar(k, n)
                nkn_bar_list.append({"graph": G, "params": {"k": k, "n": n}})
            except Exception as e:
                print(f"Error generating N_k_n_bar for k={k}, n={n}: {e}")
    instances["N_k_n_bar"] = nkn_bar_list

    return instances

def generate_extremal_instancesn(n, delta=5, num_k_samples=3):
    instances = {}
    instances["Petersen"] = [{"graph": nx.petersen_graph(), "params": {}}]
    
    chosen_n = n
    nkn_list = []
    for k in np.linspace(2, chosen_n/2, num_k_samples):
        k_int = int(round(k))
        if chosen_n < 2 * k_int:
            continue
        try:
            G = examples.N_k_n(k_int, chosen_n)
            nkn_list.append({"graph": G, "params": {"k": k_int, "n": chosen_n}})
        except Exception as e:
            print(f"error generating N_k_n for k={k_int}, n={chosen_n}: {e}")
    instances["N_k_n"] = nkn_list
    lkn_list = []
    for k in np.linspace(2, chosen_n/2, num_k_samples):
        k_int = int(round(k))
        if chosen_n < k_int + 2:
            continue
        try:
            G = examples.L_k_n(k_int, chosen_n)
            lkn_list.append({"graph": G, "params": {"k": k_int, "n": chosen_n}})
        except Exception as e:
            print(f"error generating L_k_n for k={k_int}, n={chosen_n}: {e}")
    instances["L_k_n"] = lkn_list
    lkn_bar_list = []
    for k in np.linspace(2, chosen_n/2, num_k_samples):
        k_int = int(round(k))
        if chosen_n < k_int + 2:
            continue
        try:
            G = examples.L_k_n_bar(k_int, chosen_n)
            lkn_bar_list.append({"graph": G, "params": {"k": k_int, "n": chosen_n}})
        except Exception as e:
            print(f"Error generating L_k_n_bar for k={k_int}, n={chosen_n}: {e}")
    instances["L_k_n_bar"] = lkn_bar_list
    nkn_bar_list = []
    for k in np.linspace(2, chosen_n/2, num_k_samples):
        k_int = int(round(k))
        if chosen_n < 2 * k_int + 1:
            continue
        try:
            G = examples.N_k_n_bar(k_int, chosen_n)
            nkn_bar_list.append({"graph": G, "params": {"k": k_int, "n": chosen_n}})
        except Exception as e:
            print(f"Error generating N_k_n_bar for k={k_int}, n={chosen_n}: {e}")
    instances["N_k_n_bar"] = nkn_bar_list
    
    return instances


def process_instance(instance):
    g = instance["graph"]
    res = refine.variable_orderings(g, visualisation=False)
    return res

def process_all_instances(instances):
    results_by_type = {gtype: [] for gtype in instances}
    with concurrent.futures.ProcessPoolExecutor(max_workers=2) as executor:
        futures = []
        for gtype, inst_list in instances.items():
            for inst in inst_list:
                future = executor.submit(process_instance, inst)
                futures.append((gtype, inst["params"], future))
        for gtype, params, future in futures:
            try:
                res = future.result()
                results_by_type[gtype].append({"params": params, "orderings": res})
            except Exception as e:
                print(f"Error processing {gtype} with params {params}: {e}")
    return results_by_type



def aggregate_invariants(results_by_type):
    aggregated = {}
    for gtype, results in results_by_type.items():
        for res in results:
            orderings = res["orderings"]
            for ordering_name, inv in orderings.items():
                if ordering_name not in aggregated:
                    aggregated[ordering_name] = {}
                if gtype not in aggregated[ordering_name]:
                    aggregated[ordering_name][gtype] = {"spectral_radius": [], "cheeger": []}
                aggregated[ordering_name][gtype]["spectral_radius"].append(inv["spectral_radius"])
                aggregated[ordering_name][gtype]["cheeger"].append(inv["cheeger"])
    return aggregated

def plot_aggregated_invariants(aggregated_metrics):
    strategies = list(aggregated_metrics.keys())
    graph_type_colors = {
        "Petersen": "blue",
        "N_k_n": "red",
        "L_k_n": "green",
        "L_k_n_bar": "orange",
        "N_k_n_bar": "purple"
    }
    for ordering in strategies:
        plt.figure(figsize=(10, 6))
        for gtype, metrics in aggregated_metrics[ordering].items():
            sr_vals = np.array(metrics["spectral_radius"])
            ch_vals = np.array(metrics["cheeger"])
            jitter_sr = np.random.uniform(-0.002, 0.002, size=sr_vals.shape)
            jitter_ch = np.random.uniform(-0.00015, 0.00015, size=ch_vals.shape)
            color = graph_type_colors.get(gtype, "black")
            plt.scatter(sr_vals + jitter_sr, ch_vals + jitter_ch, color=color, s=10, alpha=0.7, label=gtype)

            avg_sr = np.mean(sr_vals)
            avg_ch = np.mean(ch_vals)
            plt.axvline(x=avg_sr, color=color, linestyle='dotted', linewidth=1)
            plt.axhline(y=avg_ch, color=color, linestyle='dotted', linewidth=1)
        
        plt.xlabel("spectral radius")
        plt.ylabel("cheeger constant")
        plt.title(f"scatter for reduction with ordering: {ordering}")
        handles = []
        for gtype, col in graph_type_colors.items():
            handles.append(plt.Line2D([], [], marker='o', linestyle='', color=col, label=gtype))
        plt.legend(handles=handles, title="graph", loc="best")
        plt.grid(True)
        plt.show()


def plot_aggregated_invariants_with_cand(aggregated_metrics, cand):
    strategies = list(aggregated_metrics.keys())
    graph_type_colors = {
        "Petersen": "blue",
        "N_k_n": "red",
        "L_k_n": "green",
        "L_k_n_bar": "orange",
        "N_k_n_bar": "purple"
    }
    for ordering in strategies:
        plt.figure(figsize=(10, 6))
        for gtype, metrics in aggregated_metrics[ordering].items():
            sr_vals = np.array(metrics["spectral_radius"])
            ch_vals = np.array(metrics["cheeger"])
            jitter_sr = np.random.uniform(-0.002, 0.002, size=sr_vals.shape)
            jitter_ch = np.random.uniform(-0.00015, 0.00015, size=ch_vals.shape)
            color = graph_type_colors.get(gtype, "black")
            plt.scatter(sr_vals + jitter_sr, ch_vals + jitter_ch, color=color, s=10, alpha=0.7, label=gtype)
            plt.plot(cand[ordering]["spectral_radius"], cand[ordering]["cheeger"], color='black', label="candidate")
            avg_sr = np.mean(sr_vals)
            avg_ch = np.mean(ch_vals)
            plt.axvline(x=avg_sr, color=color, linestyle='dotted', linewidth=1)
            plt.axhline(y=avg_ch, color=color, linestyle='dotted', linewidth=1)
            plt.axvline(x=cand[ordering]["spectral_radius"], color='black', linestyle='dotted', linewidth=1)
            plt.axhline(y=cand[ordering]["cheeger"], color='black', linestyle='dotted', linewidth=1)
        
        plt.xlabel("spectral radius")
        plt.ylabel("cheeger constant")
        plt.title(f"scatter for reduction with ordering: {ordering}")
        handles = []
        for gtype, col in graph_type_colors.items():
            handles.append(plt.Line2D([], [], marker='o', linestyle='', color=col, label=gtype))
        plt.legend(handles=handles, title="graph", loc="best")
        plt.grid(True)
        plt.show()


def plot_extremal_candidate(aggregated_metrics, cand, ext_type=None):
    graph_type_colors = {
        "Petersen": "blue",
        "N_k_n": "red",
        "L_k_n": "green",
        "L_k_n_bar": "orange",
        "N_k_n_bar": "purple"
    }
    
    strategies = list(aggregated_metrics.keys())
    
    for ordering in strategies:
        plt.figure(figsize=(10, 6))
        
        if ext_type is not None:
            if ext_type in aggregated_metrics[ordering]:
                metrics = aggregated_metrics[ordering][ext_type]
                sr_vals = np.array(metrics["spectral_radius"])
                ch_vals = np.array(metrics["cheeger"])
                jitter_sr = np.random.uniform(-0.002, 0.002, size=sr_vals.shape)
                jitter_ch = np.random.uniform(-0.00015, 0.00015, size=ch_vals.shape)
                color = graph_type_colors.get(ext_type, "black")
                plt.scatter(sr_vals + jitter_sr, ch_vals + jitter_ch, color=color,
                            s=10, alpha=0.7, label=ext_type)
                avg_sr = np.mean(sr_vals)
                avg_ch = np.mean(ch_vals)
                plt.axvline(x=avg_sr, color=color, linestyle='dotted', linewidth=1)
                plt.axhline(y=avg_ch, color=color, linestyle='dotted', linewidth=1)
            else:
                print(f"ordering {ordering} does not contain data for extremal type {ext_type}.")
        else:
            for gtype, metrics in aggregated_metrics[ordering].items():
                sr_vals = np.array(metrics["spectral_radius"])
                ch_vals = np.array(metrics["cheeger"])
                jitter_sr = np.random.uniform(-0.002, 0.002, size=sr_vals.shape)
                jitter_ch = np.random.uniform(-0.00015, 0.00015, size=ch_vals.shape)
                color = graph_type_colors.get(gtype, "black")
                plt.scatter(sr_vals + jitter_sr, ch_vals + jitter_ch,
                            color=color, s=10, alpha=0.7, label=gtype)
                avg_sr = np.mean(sr_vals)
                avg_ch = np.mean(ch_vals)
                plt.axvline(x=avg_sr, color=color, linestyle='dotted', linewidth=1)
                plt.axhline(y=avg_ch, color=color, linestyle='dotted', linewidth=1)
        
        if ordering in cand:
            cand_sr = cand[ordering]["spectral_radius"]
            cand_ch = cand[ordering]["cheeger"]
            plt.scatter(cand_sr, cand_ch, color='black', marker='^', s=100, label="Candidate")
            plt.axvline(x=cand_sr, color='black', linestyle='dotted', linewidth=1)
            plt.axhline(y=cand_ch, color='black', linestyle='dotted', linewidth=1)
        
        plt.xlabel("spectral rad")
        plt.ylabel("cheeger constant")
        plt.title(f"comparison cand vs extr {ordering}")
        
        handles = []
        if ext_type is not None:
            handles.append(plt.Line2D([], [], marker='o', linestyle='', color=graph_type_colors.get(ext_type, "black"),
                                       label=ext_type))
        else:
            for gtype, col in graph_type_colors.items():
                handles.append(plt.Line2D([], [], marker='o', linestyle='', color=col, label=gtype))
        handles.append(plt.Line2D([], [], marker='^', linestyle='', color='black', label="Candidate"))
        plt.legend(handles=handles, title="Graph Type", loc="best")
        plt.grid(True)
        plt.show()
