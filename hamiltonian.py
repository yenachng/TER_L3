import networkx as nx
import numpy as np
import cluster_refinement as clrf

def cl_is_complete(G):
    cl = clrf.bondy_chvatal_closure(G)
    n = G.degree
    return cl.number_of_edges() == n*(n-1)/2

