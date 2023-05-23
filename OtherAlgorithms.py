import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score, calinski_harabasz_score
from sklearn.neighbors import NearestNeighbors
import sklearn
from Preprocessing import preprocessing
import pandas as pd


def medians_for_clusters(df):
    medians = df.groupby('label').median()
    return medians


def find_optimal_epsilon(X, min_samples, plot=True):
    neigh = NearestNeighbors(n_neighbors=min_samples)
    distances, _ = neigh.fit(X).kneighbors(X)
    k_distances = np.sort(distances[:, -1])

    # if plot:
    #     plt.plot(np.arange(len(X)), k_distances)
    #     plt.xlabel("Points")
    #     plt.ylabel("k-Distances")
    #     plt.title("k-Distance Graph")
    #     plt.show()

    differences = np.diff(k_distances)
    # so there is no RuntimeWarning: divide by zero encountered
    differences = differences[differences > 1e-7]
    diff_ratios = differences[1:] / differences[:-1]

    knee_index = np.argmax(diff_ratios) + 1
    optimal_epsilon = k_distances[knee_index]

    return optimal_epsilon


def plot_epsilon(df, min_Pts):
    minPts = min_Pts
    nbrs = sklearn.neighbors.NearestNeighbors(n_neighbors=minPts).fit(df)
    distances, indices = nbrs.kneighbors(df)
    distanceDec = sorted(distances[:, minPts - 1], reverse=True)
    fig = plt.figure(figsize=(9, 6))
    ax1 = fig.add_subplot(111)

    plt.xlabel('Indeks punktu po sortowaniu')
    plt.ylabel('Dystans od trzeciego najbliższego sąsiada')
    ax1.plot(list(range(1, df.shape[0] + 1)), distanceDec)

    plt.xscale('log')
    plt.grid(axis='y')

    plt.show()


def dbscan(df, proc):
    """
        Predicts values with dbscan algorithm
    :param proc: type of preprocessing: 'minmax', 'stand', 'norm'
    :param df: dataframe
    :return: dataframe with labels, model
    """
    minsamples = 3
    # choosing value because of plot_epsilon
    epsilon = 0.3
    if proc == 'minmax':
        epsilon = 0.5
    elif proc == 'stand':
        epsilon = 2.7
    elif proc == 'norm':
        epsilon = 0.05
    dbscan = DBSCAN(eps=epsilon, min_samples=minsamples)
    labels = dbscan.fit_predict(df)
    df['label'] = labels

    return df, dbscan


def main():
    data = pd.read_csv('train.csv')
    processing = ['stand', 'norm', 'minmax']
    # for proc in processing:
    #     df = preprocessing(data, rem_over=True, process=proc)
    #     plot_epsilon(df, 3)

    for proc in processing:
        print('-----')
        print(f'Processing: {proc}')
        df = preprocessing(data, rem_over=True, process=proc)
        df_with_labels, model = dbscan(df, proc)
        df_without_labels = df_with_labels.drop("label", axis=1)
        y = df_with_labels['label']
        print(f"silhouette score: {silhouette_score(df_without_labels, y)}")
        print(f"calinski harabasz score: {calinski_harabasz_score(df_without_labels, y)}")
        data_with_labels = data.copy()
        data_with_labels['label'] = df_with_labels['label']
        print(medians_for_clusters(data_with_labels))


if __name__ == "__main__":
    main()
