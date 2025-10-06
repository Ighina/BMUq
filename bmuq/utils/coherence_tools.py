"""
refactored_refree_metrics.py

Simplified / refactored metrics that operate on two groups of precomputed embeddings:
 - real_embeddings: numpy array (N_real x D)
 - random_embeddings: numpy array (N_rand x D)

Provided functions:
 - cosine_dispersion_group(embs)
 - average_variance_group(embs)
 - average_pairwise_similarity_group(embs)
 - arp_between_groups(real_embeddings, random_embeddings, n=1)
 - segrefree_between_groups(real_embeddings, random_embeddings, correction_factor=True, bounded=False)
 - silhouette_between_groups(real_embeddings, random_embeddings, correction_factor=True, return_sample_scores=False)

Dependencies: numpy, sklearn
"""
from typing import Tuple, Optional

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from sklearn.metrics import silhouette_samples


# ------------------------
# Basic dispersion functions (operate on a single group)
# ------------------------

def cosine_dispersion_group(embs: np.ndarray) -> float:
    """
    1 - cosine similarity between group mean and its members averaged.
    Input:
      embs: array (N x D)
    Returns:
      scalar dispersion (higher = more dispersed)
    """
    if embs.size == 0:
        raise ValueError("Empty embeddings provided.")
    mean = np.nanmean(embs, axis=0).reshape(1, -1)
    return 1.0 - float(np.nanmean(cosine_similarity(mean, embs)))


def average_variance_group(embs: np.ndarray) -> float:
    """
    L2 norm of the per-dimension standard deviation (a scalar measure of spread).
    """
    if embs.size == 0:
        raise ValueError("Empty embeddings provided.")
    return float(np.linalg.norm(np.nanstd(embs, axis=0)))


def average_pairwise_similarity_group(embs: np.ndarray) -> float:
    """
    1 - mean upper-triangle cosine similarity (excluding diagonal).
    """
    if embs.size == 0:
        raise ValueError("Empty embeddings provided.")
    S = cosine_similarity(embs)
    # take strict upper triangle
    triu = np.triu(S, k=1)
    # select nonzero entries (upper triangle zeros correspond to masked below-diagonal entries)
    vals = triu[triu != 0]
    if vals.size == 0:
        return 0.0
    return 1.0 - float(np.nanmean(vals))


# ------------------------
# Two-group evaluation functions
# ------------------------

def calculate_average_similarity(query_embedding, embeddings_array):
    """
    Computes the average cosine similarity between a single query embedding and an array of embeddings.

    Args:
        query_embedding (np.ndarray): A 1D NumPy array representing the single embedding.
        embeddings_array (np.ndarray): A 2D NumPy array where each row is an embedding to compare against.

    Returns:
        float: The average cosine similarity score. Returns 0 if the embeddings_array is empty.
    """
    # Check if the array of embeddings is empty
    if embeddings_array.shape[0] == 0:
        return 0.0

    # Reshape the query_embedding to a 2D array (1, n_features) for compatibility
    # with scikit-learn's cosine_similarity function.
    query_embedding_reshaped = query_embedding.reshape(1, -1)

    # Calculate cosine similarity between the query embedding and all embeddings in the array at once.
    # This is more efficient than iterating. The result is a 2D array of shape (1, n_embeddings).
    similarity_scores = cosine_similarity(query_embedding_reshaped, embeddings_array)

    # The result is a nested array (e.g., [[0.9, 0.2, ...]]), so we take the first element
    # to get a 1D array of scores, then compute the mean.
    average_score = np.mean(similarity_scores[0])

    return average_score

def arp_between_groups(real_embeddings: np.ndarray,
                       random_embeddings: np.ndarray,
                       dispersion_fn=average_variance_group,
                       n: int = 1) -> float:
    """
    Simplified ARP-like score computed between two groups:
      - intra = dispersion(real_embeddings)
      - inter  = dispersion(concatenation(real_embeddings, random_embeddings))
      - score = (inter**n - intra**n) / (inter**n + intra**n)
    Output in [-1, 1] (higher means inter >> intra).
    """
    if real_embeddings.size == 0:
        raise ValueError("real_embeddings must be non-empty.")
    # compute intra
    intra = float(dispersion_fn(real_embeddings))
    inter = float(dispersion_fn(random_embeddings))

    # safeguard against zero denominators
    denom = (inter**n + intra**n)
    if denom == 0:
        return 0.0
    return float((inter**n - intra**n) / denom)


def segrefree_between_groups(real_embeddings: np.ndarray,
                             random_embeddings: np.ndarray,
                             correction_factor: bool = True,
                             bounded: bool = False,
                             negative: bool = False) -> Optional[float]:
    """
    Simplified SegReFree-style ratio between two groups.
      - Compute S_real = mean euclidean distance of members in `real_embeddings` to its centroid.
      - Compute S_rand = mean euclidean distance of members in `random_embeddings` to its centroid.
      - Compute centroid distance D = euclidean_distance(mean_real, mean_rand)
      - R = (S_real + S_rand) / D    (or bounded version = (S_real + S_rand)/(D + (S_real+S_rand)))
      - Optionally apply a correction factor for small groups:
          multiply numerator by 1 / (1 - 1/sqrt(len_group)) (applied separately to each group's S)
    Returns:
      float R (or None if one of the groups has size 0 or centroids coincide and denom=0).
    """
    if real_embeddings.size == 0 or random_embeddings.size == 0:
        raise ValueError("Both real_embeddings and random_embeddings must be non-empty.")

    n_real = real_embeddings.shape[0]
    n_rand = random_embeddings.shape[0]

    # centroids
    mean_real = np.nanmean(real_embeddings, axis=0).reshape(1, -1)
    mean_rand = np.nanmean(random_embeddings, axis=0).reshape(1, -1)

    # mean distances to centroid
    S_real = float(np.nanmean(euclidean_distances(mean_real, real_embeddings)))
    S_rand = float(np.nanmean(euclidean_distances(mean_rand, random_embeddings)))

    if correction_factor:
        # original code used 1 - 1/len(segment) ; emulate a de-biasing by scaling S values
        # use safe guards for len=1
        cf_real = 1.0 - (1.0 / n_real) if n_real > 1 else 0.0
        cf_rand = 1.0 - (1.0 / n_rand) if n_rand > 1 else 0.0
        # avoid dividing by zero; if cf is 0 (len==1) keep original S
        if cf_real > 0:
            S_real = S_real / cf_real
        if cf_rand > 0:
            S_rand = S_rand / cf_rand

    numerator = S_real + S_rand
    D = float(euclidean_distances(mean_real, mean_rand)[0, 0])

    if bounded:
        denom = D + numerator
    else:
        denom = D

    if denom == 0:
        return None  # cannot compute (coincident centroids and non-bounded) or both zero
    result = numerator / denom
    if negative:
        result = -result
    return float(result)


def silhouette_between_groups(real_embeddings: np.ndarray,
                              random_embeddings: np.ndarray,
                              correction_factor: bool = True) -> Tuple[float, Optional[np.ndarray]]:
    """
    Compute silhouette score for a 2-cluster setting where:
      - cluster 0 = real_embeddings
      - cluster 1 = random_embeddings
    Returns:
      mean_silhouette

    If correction_factor True, multiply each sample's silhouette by cf = (1 - 1/cluster_size)
    (mirrors earlier behaviour that reduces score contribution from clusters of size 1).
    """
    if real_embeddings.size == 0 or random_embeddings.size == 0:
        raise ValueError("Both real_embeddings and random_embeddings must be non-empty.")

    X = np.vstack([real_embeddings, random_embeddings])
    labels = np.array([0] * real_embeddings.shape[0] + [1] * random_embeddings.shape[0])

    # sklearn silhouette requires at least 2 samples per cluster to produce meaningful sample scores;
    # silhouette_score will still run but silhouette_samples may produce -1 or near-zero for tiny clusters.
    # We'll compute silhouette_samples and take mean.
    sample_scores = silhouette_samples(X, labels, metric='euclidean')

    if correction_factor:
        # apply per-sample correction factor based on cluster size
        size0 = np.sum(labels == 0)
        size1 = np.sum(labels == 1)
        cf0 = 1.0 - (1.0 / size0) if size0 > 1 else 0.0
        cf1 = 1.0 - (1.0 / size1) if size1 > 1 else 0.0
        # multiply each sample by its cluster correction factor
        sample_scores[:size0] = sample_scores[:size0] * cf0
        sample_scores[size0:] = sample_scores[size0:] * cf1

    mean_score = float(np.mean(sample_scores))
    
    return mean_score


# ------------------------
# Small example usage (not executed)
# ------------------------
if __name__ == "__main__":
    # Example: two toy groups of embeddings (N x D)
    import numpy as np
    rng = np.random.RandomState(0)
    real = rng.normal(loc=0.0, scale=1.0, size=(50, 768))
    rand = rng.normal(loc=0.1, scale=1.1, size=(50, 768))

    print("ARP between groups:", arp_between_groups(real, rand))
    print("SegReFree between groups:", segrefree_between_groups(real, rand))
    sil_mean, sil_samples = silhouette_between_groups(real, rand, return_sample_scores=True)
    print("Silhouette mean:", sil_mean)