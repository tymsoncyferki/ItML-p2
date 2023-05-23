import numpy as np
from scipy.spatial import distance
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from kmodes.kprototypes import KPrototypes
import matplotlib.pyplot as plt


def num_of_clus_elbow(data) -> int:
    """
        Finds optimal number of clusters using elbow method.

        :param data: dataframe
        :return: number of clusters
    """
    sum_of_squared_distances = []
    K = range(1, 10)
    for num_clusters in K:
        kmeans = KMeans(n_clusters=num_clusters)
        kmeans.fit(data)
        sum_of_squared_distances.append(kmeans.inertia_)

    # finding elbow point
    sd = np.diff(sum_of_squared_distances, 2)
    elbow_point = np.argmax(sd) + 2

    # plt.figure(figsize=(8, 5))
    # plt.plot(range(1, 10), sum_of_squared_distances, marker='o')
    # plt.xlabel('Number of clusters')
    # plt.ylabel('WCSS')
    # plt.show()

    return elbow_point


def num_of_clus_silhouette(data) -> int:
    """
    Finds optimal number of clusters using silhouette method.

    :param data: dataframe
    :return: number of clusters
    """
    silhouette_scores = []
    for i in range(2, 11):
        kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=0)
        kmeans.fit(data)
        score = silhouette_score(data, kmeans.labels_)
        silhouette_scores.append(score)

    silhouette_point = np.argmax(silhouette_scores) + 2

    # plt.figure(figsize=(8, 5))
    # plt.plot(range(1, 10), silhouette_scores, marker='o')
    # plt.xlabel('Number of clusters')
    # plt.ylabel('Silhouette score')
    # plt.show()

    return silhouette_point


def k_means_alg(df, meth_num_clus="silhouette", num_clusters=None) -> list[int]:
    """
    Finds labels

    :param df: dataframe
    :param meth_num_clus:  Determines number of clusters:
        "silhouette": silhouette method
        "elbow": elbow method
    :param num_clusters: Specifies number of clusters
    :return: dataframe with labels, model
    """
    if not num_clusters:
        if meth_num_clus == "elbow":
            num_clusters = num_of_clus_elbow(df)
        else:
            num_clusters = num_of_clus_silhouette(df)
    # num_clusters = 4
    print(f'number of clusters: {num_clusters}')
    kmeans = KMeans(n_clusters=num_clusters, init='k-means++', max_iter=300, n_init=10, random_state=0)
    y_kmeans = kmeans.fit_predict(df)
    df["label"] = y_kmeans

    return df, kmeans


def k_prototypes_alg(df, meth_num_clus="silhouette", num_clusters=None):
    if not num_clusters:
        if meth_num_clus == "elbow":
            num_clusters = num_of_clus_elbow(df)
        else:
            num_clusters = num_of_clus_silhouette(df)
    # num_clusters = 4
    print(f'number of clusters: {num_clusters}')
    kp = KPrototypes(n_clusters=num_clusters, max_iter=300, n_init=10, random_state=0)
    y_kp = kp.fit_predict(df, categorical=[1, 2, 5, 6, 8, 10])
    df["label"] = y_kp

    return df, kp


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

