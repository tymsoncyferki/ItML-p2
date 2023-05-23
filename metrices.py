import numpy as np
from scipy.spatial import distance
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score


def min_interclust_dist(X, label):
    clusters = set(label)
    global_min_dist = np.inf
    for cluster_i in clusters:
        cluster_i_idx = np.where(label == cluster_i)
        for cluster_j in clusters:
            if cluster_i != cluster_j:
                cluster_j_idx = np.where(label == cluster_j)
                interclust_min_dist = np.min(distance.cdist(X.iloc[cluster_i_idx], X.iloc[cluster_j_idx]))
                global_min_dist = np.min([global_min_dist, interclust_min_dist])
    return global_min_dist


def _inclust_mean_dists(X, label):
    clusters = set(label)
    inclust_dist_list = []
    for cluster_i in clusters:
        cluster_i_idx = np.where(label == cluster_i)
        inclust_dist = np.mean(distance.pdist(X.iloc[cluster_i_idx]))
        inclust_dist_list.append(inclust_dist)
    return inclust_dist_list


def mean_inclust_dist(X, label):
    inclust_dist_list = _inclust_mean_dists(X, label)
    return np.mean(inclust_dist_list)


def std_dev_of_inclust_dist(X, label):
    inclust_dist_list = _inclust_mean_dists(X, label)
    return np.std(inclust_dist_list)


# def mean_dist_to_center(X, label):
#     clusters = set(label)
#     inclust_dist_list = []
#     for cluster_i in clusters:
#         cluster_i_idx = np.where(label == cluster_i)
#         cluster_i_mean = np.mean(X.iloc[cluster_i_idx], axis=0)
#         inclust_dist = np.mean(distance.cdist(X.iloc[cluster_i_idx], cluster_i_mean))
#         inclust_dist_list.append(inclust_dist)
#     return np.mean(inclust_dist_list)


metrices = {
    'Minimal distance between clusters': min_interclust_dist,
    'Average distance between points in the same class': mean_inclust_dist,
    'Standard deviation of distance between points in the same class': std_dev_of_inclust_dist,
    # 'Average distance to cluster center': mean_dist_to_center,
    'Silhouette score': silhouette_score,
    'Calinski-Harabasz index': calinski_harabasz_score,
    'Davies Bouldin index': davies_bouldin_score
}
