import networkx as nx
import numpy as np
import scipy.sparse as sps
from scipy.sparse.linalg import eigsh
from sklearn.cluster import KMeans

def computeA(G):
    A_sparse = nx.adjacency_matrix(G)
    diags = np.array(A_sparse.sum(axis=1)).flatten()
    D = sps.diags(diags, 0)
    return A_sparse

def compute_normalized(G):
    A = computeA(G)
    diags = np.array(A.sum(axis=1)).flatten()
    n = A.shape[0]
    D_inv_sqrt = sps.diags(1/np.sqrt(diags+1e-10), 0) #in case disconnected
    I = sps.diags(np.ones(n), 0)
    L_norm = I - D_inv_sqrt@A@D_inv_sqrt
    Q_norm = I + D_inv_sqrt@A@D_inv_sqrt
    return L_norm, Q_norm, n, A

def spectral_radius_power(G, max_iter=1000, tol=1e-6):
    A = nx.adjacency_matrix(G)
    n = A.shape[0]
    x = np.random.rand(n)
    x /= np.linalg.norm(x)
    for _ in range(max_iter):
        x_new = A @ x
        x_new_norm = np.linalg.norm(x_new)
        x_new /= x_new_norm

        if np.linalg.norm(x - x_new) < tol:
            break
        x = x_new
    return x_new_norm

def projection_lower(A, k, p=10, q=2, random_state = None):
    n = A.shape[0]
    if random_state is not None:
        rng = np.random.RandomState(random_state) if isinstance(random_state, int) else random_state
    else:
        rng = np.random
    O = rng.randn(n,k+p)
    Y = A.dot(O) if hasattr(A, "dot") else np.dot(A, O)
    for _ in range(q):
        Y = A.dot(Y) if hasattr(A, "dot") else np.dot(A,O)

    Q, _ = np.linalg.qr(Y, mode='reduced')
    B = Q.T.dot(A.dot(Q)) if hasattr(A,"dot") else np.dot(Q.T, np.dot(A,Q))
    evals, evecs_B = np.linalg.eigh(B)
    idx = np.argsort(evals)
    evals = evals[idx]
    evecs_B = evecs_B[:,idx]
    evecs = Q.dot(evecs_B[:,:k])
    evals = evals[:k]
    return evals, evecs

def eigengaps(eigenvals, method):
    if method=="L":
        print("calculating gaps between normalized laplacian eigenvalues")
        tol_multiplier = 1.2
    else:
        print("calculating gaps between normalized signless laplacian eigenvalues")
        tol_multiplier = 1.5
    gaps = np.diff(eigenvals)
    gaps_std = np.std(gaps)
    m_gap_indx = np.argmax(gaps)
    m_gap = gaps[m_gap_indx]
    print("std gaps",gaps_std)
    print("max gap", m_gap)
    similar_gaps = [i for i, gap in enumerate(gaps) if m_gap - gap < gaps_std * tol_multiplier]
    print("number of similar gaps, including largest", len(similar_gaps))
    return m_gap_indx, similar_gaps

def fast_eigen_decomp(G, sizetol):
    data = {}
    L_norm, Q_norm, n, A= compute_normalized(G)
    data["size"] = n
    if n > sizetol:
        k = int(n / 20)
        print(f"graph size {n} > {sizetol}: using k = {k} with QR-based approximation")
        eigvals_L, eigvects_L = projection_lower(L_norm, k, p=10, q=2, random_state=42)
        eigvals_Q, eigvects_Q = projection_lower(Q_norm, k, p=10, q=2, random_state=42)
        eigvals_Q = eigvals_Q[::-1]
        eigvects_Q = eigvects_Q[:, ::-1]
    else:
        k = 20
        print(f"graph size {n} <= {sizetol}: using full dense eigen-decomposition with k = {k}")
        L_dense = L_norm.toarray() if hasattr(L_norm, "toarray") else L_norm
        Q_dense = Q_norm.toarray() if hasattr(Q_norm, "toarray") else Q_norm
        eigvals_L, eigvects_L = np.linalg.eigh(L_dense)
        eigvals_Q, eigvects_Q = np.linalg.eigh(Q_dense)
        idx_desc = np.argsort(eigvals_Q)[::-1]
        eigvals_Q = eigvals_Q[idx_desc]
        eigvects_Q = eigvects_Q[:, idx_desc]

    max_idx_L, large_gaps_L = eigengaps(eigvals_L[1:], "L")
    _, large_gaps_Q = eigengaps(eigvals_Q, "Q")
    #check importance of max Q
    kL =len(large_gaps_L)
    kQ = len(large_gaps_Q)
    if n > sizetol:
        lam, _ = eigsh(A, k=1, which='LM', tol=1e-4, maxiter=300)
        spectral_radius = np.real(lam[0])
    else:
        A_dense = A.toarray() if hasattr(A, "toarray") else A
        lam = np.linalg.eigvals(A_dense)
        spectral_radius = np.max(np.abs(lam))
    data["spectral_radius"] = spectral_radius
    data["Perron cluster"] = eigvects_Q[:,-1].reshape(-1,1)
    if kL == 1:
        kL_used = max_idx_L + 1
        data["L_vects"] = eigvects_L[:, 1:kL_used]
        data["partitions_L"] = kL_used    
    else:
        data["L_vects"] = eigvects_L[:, 1].reshape(-1, 1)
        data["partitions_L"] = None
    
    if kQ > k * 0.85:
        data["partitions_Q"] = None
    else:
        data["partitions_Q"] = kQ
        data["Perron cluster"] = eigvects_Q[:, :-kQ]


def kmeans_clusters(embedding, k):
    if k is None:
        print("input error for k, returning None")
        return None
    kmeans = KMeans(n_clusters=k, random_state=0)
    labels = kmeans.fit_predict(embedding)
    return labels

def get_partitions(data):

    if data["partitions_L"] is not None:
        k = data["partitions_L"]
        print(f"found clustering behavior on laplacian eigenvalues for {k} clusters")
        embedding = data["L_vects"]

    elif data["partitions_Q"] is not None:
        k = data["partitions_Q"]
        print(f"found clustering behavior on signless laplacian eigenvalues for {k} clusters")
        embedding = data["Perron cluster"]

    else:
        k = 2
        print(f"couldn't find any clear gaps on laplacian or signless laplacian eigenvalues, partitioning using the fiedler vector")
        embedding = data["L_vects"]

    labels = kmeans_clusters(embedding, k)
    
    return labels


