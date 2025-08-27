import numpy as np
from numba import njit


@njit
def _pairwise_distances(X):
    """
    Compute pairwise Euclidean distances between rows in X.
    
    Parameters
    ----------
    X : np.ndarray
        2D array where each row is a data point.

    Returns
    -------
    2D array of pairwise distances.
    """
    n, d = X.shape
    D = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            dist = 0.0
            for k in range(d):
                diff = X[i, k] - X[j, k]
                dist += diff * diff
            dist = np.sqrt(dist)
            D[i, j] = dist
            D[j, i] = dist
    return D

@njit
def _kmeans_numba(X, random_seed:int, k=2, max_iters=10):
    """Perform k-means clustering on data X.
    Parameters
    ----------
    X : np.ndarray
        2D array where each row is a data point.
    k : int
        Number of clusters.
    max_iters : int
        Maximum number of iterations.
    
    Returns
    -------
    Assignments for each data point to clusters and final centroids. 
    """
    n, d = X.shape
    np.random.seed(random_seed)
    #centroids = X[np.random.choice(n, k, replace=False)]
    labels = np.zeros(n, dtype=np.int64)
    
    chosen = np.zeros(n, dtype=np.int64)
    centroids = np.zeros((k, d))
    for j in range(k):
        while True:
            idx = np.random.randint(0, n)
            if chosen[idx] == 0:
                chosen[idx] = 1
                centroids[j] = X[idx]
                break
    
    for _ in range(max_iters):
        # assign step
        for i in range(n):
            best_dist = 1e18
            best_idx = 0
            for j in range(k):
                dist = 0.0
                for dim in range(d):
                    diff = X[i, dim] - centroids[j, dim]
                    dist += diff * diff
                if dist < best_dist:
                    best_dist = dist
                    best_idx = j
            labels[i] = best_idx

        # update step
        new_centroids = np.zeros((k, d))
        counts = np.zeros(k, dtype=np.int64)
        for i in range(n):
            new_centroids[labels[i]] += X[i]
            counts[labels[i]] += 1
        for j in range(k):
            if counts[j] > 0:
                new_centroids[j] /= counts[j]
            else:
                new_centroids[j] = X[np.random.randint(0, n)]
        if np.allclose(new_centroids, centroids):
            break
        
        centroids = new_centroids

    return centroids, labels

@njit
def _spectral_clustering_numba(clus_data, taxon_threshold, random_seed:int, 
                                split_size: int = 10):
    """ Perform spectral clustering on the given data.
    Parameters
    ----------
    clus_data : np.ndarray
        2D array where each row is a data point.
    taxon_threshold : float
            Threshold for Gaussian similarity.
    random_seed : int
        Random seed for k-means initialization.
    split_size : int
        Minimum cluster size to consider splitting.
    
    Returns
    -------
    np.ndarray with cluster assignments for each data point.
    """
    n = clus_data.shape[0]
    labels = np.zeros(n, dtype=np.int64)

    if n > split_size:
        # distance matrix
        D2Mat = _pairwise_distances(clus_data)

        # Gaussian similarity
        W = np.exp(-np.abs(D2Mat) ** 2 / (2 * taxon_threshold ** 2))
        for i in range(n):
            for j in range(n):
                if D2Mat[i, j] > taxon_threshold:
                    W[i, j] = 0.0

        # Degree matrix
        D = np.zeros((n, n))
        for i in range(n):
            D[i, i] = np.sum(W[i])

        # Laplacian
        L = D - W

        # Eigen-decomposition
        E, U = np.linalg.eigh(L)
        for i in range(len(E)):
            if np.abs(E[i]) < 1e-10:
                E[i] = 0.0

        n_comp = np.sum(E == 0)
        if n_comp > 1:
            k = 2 if n_comp > 1 else 1
            _, labels = _kmeans_numba(U[:, :k], random_seed, k=k)

            # enforce cluster size constraints
            count1 = np.sum(labels == 0)
            count2 = np.sum(labels == 1)
            if count1 < split_size // 2:
                for i in range(n):
                    if labels[i] == 0:
                        labels[i] = 1
            if count2 < split_size // 2:
                for i in range(n):
                    if labels[i] == 1:
                        labels[i] = 0
            if np.all(labels == 1):
                labels = np.zeros(n, dtype=np.int64)

    return labels


@njit
def _compute_taxon_ids_numba(current_ancestor_id: np.ndarray, 
                             individuals_trait: np.ndarray,
                             individuals_taxon_id: np.ndarray,
                             taxon_threshold: float,
                             random_seed: int = 0
                             ):
    
    """ Compute taxon IDs based on traits and ancestor IDs.
    Parameters
    ----------
    current_ancestor_id : np.ndarray
        Array of current ancestor IDs.
    individuals_trait : np.ndarray
        2D array where each row is the trait vector of an individual.
    individuals_taxon_id : np.ndarray
        Array of current taxon IDs.
    taxon_threshold : float
        Threshold for clustering.
    random_seed : int
        Random seed for reproducibility.
    
    Returns
    -------
    new_taxon_id as an np.ndarray with updated taxon IDs after clustering and
    current_ancestor_id as an np.ndarray with Uúnchanged current ancestor IDs.
    """

    max_clus = np.int64(np.max(individuals_taxon_id))
    new_taxon_id = np.zeros_like(current_ancestor_id)

    unique_ans = np.unique(current_ancestor_id)

    for ans in unique_ans:
        ans_indx = np.where(current_ancestor_id == ans)[0]

        clusdata = individuals_trait[ans_indx]

        # Run spectral clustering
        clus = _spectral_clustering_numba(clusdata, taxon_threshold, random_seed=random_seed)

        # Adjust cluster IDs relative to max_clus
        if max_clus < np.max(clus):
            new_clus = clus + 1
            for i in range(len(ans_indx)):
                new_taxon_id[ans_indx[i]] = new_clus[i]
        else:
            new_clus = clus + 1 + max_clus
            for i in range(len(ans_indx)):
                new_taxon_id[ans_indx[i]] = new_clus[i]

        max_clus = np.int64(np.max(new_clus))

    return new_taxon_id, current_ancestor_id

@njit
def _points_in_polygon(points, poly):
    """
    Vectorized ray casting algorithm for multiple points.
    Parameters
    ----------
    points : np.ndarray
        2D array where each row is a point (x, y).
    poly : np.ndarray
        2D array where each row is a vertex of the polygon (x, y).
    
    Returns
    -------
    np.ndarray with boolean array indicating if each point is inside the polygon.
        """
    n_points = points.shape[0]
    n_poly = poly.shape[0]
    inside = np.zeros(n_points, dtype=np.bool_)

    j = n_poly - 1
    for i in range(n_poly):
        xi, yi = poly[i, 0], poly[i, 1]
        xj, yj = poly[j, 0], poly[j, 1]

        for p in range(n_points):
            px, py = points[p, 0], points[p, 1]
            # Check if horizontal ray crosses polygon edge
            if ((yi > py) != (yj > py)) and (px < (xj - xi) * (py - yi) / (yj - yi + 1e-12) + xi):
                inside[p] = not inside[p]
        j = i

    return inside
