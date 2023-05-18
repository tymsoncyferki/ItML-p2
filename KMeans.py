import numpy as np
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
    :return: List with labels
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
    #  df["label"] = y_kmeans

    return y_kmeans


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
    return y_kp

