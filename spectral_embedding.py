import networkx as nx
import numpy as np
import scipy.sparse as sps
from scipy.sparse.linalg import eigsh
from sklearn.cluster import KMeans
import cluster_refinement as refine

def to_adjacency(G_or_A):
    if isinstance(G_or_A, nx.Graph):
        return nx.adjacency_matrix(G_or_A)
    elif isinstance(G_or_A, (np.ndarray, sps.spmatrix)):
        return G_or_A
    else:
        raise ValueError("input type error")

def QR_orthogonalization(A, k, p=10, q=2,):
    n = A.shape[0]
    O = np.random(n,k+p)
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

def spectral_gap(G):
    A = to_adjacency(G)
    eigvals, _ = eigsh(A, k=2,which="LA")
    sorted = np.sort(eigvals)[::-1]
    return eigvals[0]-eigvals[1]


def power_method(A, tol=1e-6, max_iter=1000, x0=None):
    n = A.shape[0]
    if x0 is None or np.allclose(x0, 0):
        x = np.ones(n)
    else:
        x = x0
    x = x / np.linalg.norm(x)
    lam_old = 0
    for i in range(max_iter):
        x = A.dot(x)
        norm = np.linalg.norm(x)
        if norm == 0:
            return 0, x, i
        x = x / norm
        lam = np.dot(x, A.dot(x))
        if abs(lam - lam_old) < tol:
            return lam, x, i+1
        lam_old = lam
    return lam, x, max_iter

def spectral_radius(A):
    A=to_adjacency(A)
    lam, _, _= power_method(A)
    return lam

def compute_normalized(G_or_A):
    A = to_adjacency(G_or_A)
    if sps.issparse(A):
        diags = np.array(A.sum(axis=1)).flatten()
    else:
        diags = np.sum(A, axis=1)
    with np.errstate(divide='ignore'):
        inv_sqrt = 1 / np.sqrt(diags)
    inv_sqrt[np.isinf(inv_sqrt)] = 0.0
    D_inv = sps.diags(inv_sqrt, format='csr')
    n = A.shape[0]
    I = sps.eye(n, format='csr')
    L_norm = I - D_inv @ A @ D_inv
    Q_norm = I + D_inv @ A @ D_inv
    return L_norm, Q_norm, n, A

def projection_lower(A, k, p=10, q=2, random_state=None):
    n = A.shape[0]
    rng = np.random.RandomState(random_state) if isinstance(random_state, int) else (random_state or np.random)
    O = rng.randn(n, k+p)
    Y = A.dot(O) if hasattr(A, "dot") else np.dot(A, O)
    for _ in range(q):
        Y = A.dot(Y) if hasattr(A, "dot") else np.dot(A, Y)
    Q, _ = np.linalg.qr(Y, mode='reduced')
    B = Q.T.dot(A.dot(Q)) if hasattr(A, "dot") else np.dot(Q.T, np.dot(A, Q))
    evals, evecs_B = np.linalg.eigh(B)
    idx = np.argsort(evals)
    evals = evals[idx][:k]
    evecs_B = evecs_B[:, idx][:, :k]
    evecs = Q.dot(evecs_B)
    return evals, evecs

def eigengaps(eigenvals, method="L"):
    tol = 1.2 if method=="L" else 1.5
    gaps = np.diff(eigenvals)
    if len(gaps)==0:
        return 0, []
    std_gap = np.std(gaps)
    max_idx = np.argmax(gaps)
    m_gap = gaps[max_idx]
    similar = [i for i, gap in enumerate(gaps) if (m_gap - gap) < std_gap * tol]
    return max_idx, similar

def fast_eigen_decomp(G_or_A, sizetol=750):
    data = {}
    L_norm, Q_norm, n, A = compute_normalized(G_or_A)
    data["size"] = n
    if n > sizetol:
        k = int(max(20, np.ceil(10 * np.log(n))))
        print(f"graph size {n} > {sizetol}: using k = {k} with QR approximation")
        Ld = L_norm.toarray() if sps.issparse(L_norm) else L_norm
        Qd = Q_norm.toarray() if sps.issparse(Q_norm) else Q_norm
        evals_L, evecs_L = projection_lower(Ld, k, p=10, q=2, random_state=42)
        evals_Q, evecs_Q = projection_lower(Qd, k, p=10, q=2, random_state=42)
        evals_Q = evals_Q[::-1]
        evecs_Q = evecs_Q[:, ::-1]
    else:
        k = 20
        print(f"graph size {n} <= {sizetol}: using full dense eigendecomposition with k = {k}")
        Ld = L_norm.toarray() if sps.issparse(L_norm) else L_norm
        Qd = Q_norm.toarray() if sps.issparse(Q_norm) else Q_norm
        evals_L, evecs_L = np.linalg.eigh(Ld)
        evals_Q, evecs_Q = np.linalg.eigh(Qd)
        idx_desc = np.argsort(evals_Q)[::-1]
        evals_Q = evals_Q[idx_desc]
        evecs_Q = evecs_Q[:, idx_desc]
    if len(evals_L) > 1:
        max_idx_L, gaps_L = eigengaps(evals_L[1:], "L")
    else:
        max_idx_L, gaps_L = 0, []
    _, gaps_Q = eigengaps(evals_Q, "Q")
    kL = len(gaps_L)
    kQ = len(gaps_Q)
    if n > sizetol:
        try:
            lam, _ = eigsh(A, k=1, which='LM', tol=1e-4, maxiter=300)
            sr = np.real(lam[0])
        except Exception:
            Ad = A.toarray() if sps.issparse(A) else A
            lam = np.linalg.eigvals(Ad)
            sr = np.max(np.abs(lam))
    else:
        Ad = A.toarray() if sps.issparse(A) else A
        lam = np.linalg.eigvals(Ad)
        sr = np.max(np.abs(lam))
    data["spectral_radius"] = sr
    data["perron_cluster"] = evecs_Q[:, -1].reshape(-1, 1)
    if kL == 1 and max_idx_L != 0:
        kL_used = max_idx_L + 1
        data["L_vects"] = evecs_L[:, 1:kL_used]
        data["partitions_L"] = kL_used
    else:
        data["L_vects"] = evecs_L[:, 1].reshape(-1, 1)
        data["partitions_L"] = None
    data["partitions_Q"] = None if kQ > k * 0.85 else kQ
    if kQ <= k * 0.85:
        data["perron_cluster"] = evecs_Q[:, :-kQ]
    data["fiedler_vector"] = evecs_L[:, 1].reshape(-1, 1) if evecs_L.shape[1] > 1 else evecs_L.reshape(-1, 1)
    return data

def kmeans_clusters(embedding, k):
    if k is None or embedding is None:
        return None
    kmeans = KMeans(n_clusters=k, random_state=0)
    return kmeans.fit_predict(embedding)

def get_partitions(data):

    if data["partitions_L"] is not None:
        k = data["partitions_L"]
        print(f"found clustering behavior on laplacian eigenvalues for {k} clusters")
        embedding = data["L_vects"]

    elif data["partitions_Q"] is not None:
        k = data["partitions_Q"]
        print(f"found clustering behavior on signless laplacian eigenvalues for {k} clusters")
        embedding = data["perron_cluster"]

    else:
        k = 2
        print(f"couldn't find any clear gaps on laplacian or signless laplacian eigenvalues, partitioning using the fiedler vector")
        embedding = data["L_vects"]

    labels = kmeans_clusters(embedding, k)
    
    return labels

def largest_clique(G):
    cliques = list(nx.find_cliques(G))
    if not cliques:
        return set()
    max_size = max(len(c) for c in cliques)
    for clique in cliques:
        if len(clique) == max_size:
            return set(clique)
    return set()