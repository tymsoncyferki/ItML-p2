import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


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
    return silhouette_point


def k_means_alg(df, meth_num_clus="silhouette") -> list[int]:
    """
    Finds labels

    :param df: dataframe
    :param meth_num_clus:  Determines number of clusters:
        "silhouette": silhouette method
        "elbow": elbow method
    :return: List with labels
    """
    if meth_num_clus == "elbow":
        num_clusters = num_of_clus_elbow(df)
    else:
        num_clusters = num_of_clus_silhouette(df)

    kmeans = KMeans(n_clusters=num_clusters, init='k-means++', max_iter=300, n_init=10, random_state=0)
    y_kmeans = kmeans.fit_predict(df)
    #  df["label"] = y_kmeans

    return y_kmeans
