from sklearn_extra.cluster import KMedoids
from KMeans import num_of_clus_elbow, num_of_clus_silhouette
from sklearn.cluster import AgglomerativeClustering, MiniBatchKMeans
from sklearn.mixture import GaussianMixture


def k_medoids_alg(df, meth_num_clus="silhouette", num_clusters=None):
    if not num_clusters:
        if meth_num_clus == "elbow":
            num_clusters = num_of_clus_elbow(df)
        else:
            num_clusters = num_of_clus_silhouette(df)

    print(f'Number of clusters: {num_clusters}')
    kmedoids = KMedoids(n_clusters=num_clusters, random_state=0)
    kmedoids.fit(df)
    y_kmedoids = kmedoids.predict(df)
    df = df.copy()
    df["label"] = y_kmedoids

    return df, kmedoids


def aglomerative_alg(df, num_clusters, linkage):

    print(f'Number of clusters: {num_clusters}')
    aglinks = AgglomerativeClustering(n_clusters=num_clusters, linkage=linkage)
    y_aglinks = aglinks.fit_predict(df)
    df = df.copy()
    df["label"] = y_aglinks

    return df, aglinks


def minibatch_alg(df, meth_num_clus="silhouette", num_clusters=None):
    if not num_clusters:
        if meth_num_clus == "elbow":
            num_clusters = num_of_clus_elbow(df)
        else:
            num_clusters = num_of_clus_silhouette(df)

    miniBatchKmeans = MiniBatchKMeans(n_clusters=num_clusters)
    y_minibatch = miniBatchKmeans.fit_predict(df)
    df = df.copy()
    df["label"] = y_minibatch

    return df, miniBatchKmeans


def gmm_alg(df, n_components, covariance_type='full'):

    gmm = GaussianMixture(n_components=n_components, covariance_type=covariance_type)
    y_gmm = gmm.fit_predict(df)
    df = df.copy()
    df['label'] = y_gmm

    return df, y_gmm

