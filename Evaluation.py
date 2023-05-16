from sklearn.metrics import silhouette_score, calinski_harabasz_score, silhouette_samples
import matplotlib.pyplot as plt
from preprocessing import preprocessing
from KMeans import k_means_alg, k_prototypes_alg
import numpy as np
import pandas as pd


def main():
    # reading and preprocessing dataframe
    data = pd.read_csv('heart_disease_patients.csv')

    # print('Baseline model:')
    # df = preprocessing(data, rem_over=False, process='stand')
    # # applying algorithm
    # y_kmeans = k_means_alg(df)
    # # some scores
    # print(f"silhouette score: {silhouette_score(df,y_kmeans)}")
    # print(f"calinski harabasz score: {calinski_harabasz_score(df, y_kmeans)}")

    processing = ['stand', 'minmax', 'norm']
    algorithms = [(k_means_alg, 'K Means'), (k_prototypes_alg, 'K Prototypes')]
    for alg, name in algorithms:
        print('-----')
        print(name)
        for proc in processing:
            print(f'Processing: {proc}')
            df = preprocessing(data, rem_over=False, process=proc)
            y = alg(df)
            print(f"silhouette score: {silhouette_score(df, y)}")
            print(f"calinski harabasz score: {calinski_harabasz_score(df, y)}")
            plot_silhouette(df, y)


def plot_silhouette(df, y_kmeans):
    # Compute silhouette scores for each sample
    silhouette_avg = silhouette_score(df, y_kmeans)
    sample_silhouette_values = silhouette_samples(df, y_kmeans)
    n = len(np.unique(y_kmeans))
    # Plot silhouette plot for 2 clusters
    fig, ax = plt.subplots()
    y_lower = 10
    for i in range(n):
        # Aggregate the silhouette scores for samples belonging to cluster i
        ith_cluster_silhouette_values = \
            sample_silhouette_values[y_kmeans == i]
        ith_cluster_silhouette_values.sort()
        size_cluster_i = ith_cluster_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster_i

        color = plt.cm.Spectral(float(i) / n)
        ax.fill_betweenx(np.arange(y_lower, y_upper),
                        0, ith_cluster_silhouette_values,
                        facecolor=color, edgecolor=color, alpha=0.7)

        # Label silhouette plots with their cluster numbers at the middle
        ax.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))
        
        # Compute the new y_lower for next plot
        y_lower = y_upper + 10

    ax.set_title(f"Silhouette plot for {n} clusters")
    ax.set_xlabel("Silhouette coefficient values")
    ax.set_ylabel("Cluster labels")

    # Add vertical line for average silhouette score
    ax.axvline(x=silhouette_avg, color="red", linestyle="--")

    # Set the x-axis limits and ticks
    ax.set_xlim([-0.1, 1])
    ax.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])

    # Set the y-axis limits and ticks
    ax.set_ylim([0, y_upper + 10])
    ax.set_yticks([])

    plt.show()

    
# # ploting inertia
# plt.plot(K,Sum_of_squared_distances,'bx-')
# plt.xlabel('Values of K') 
# plt.ylabel('Sum of squared distances') 
# plt.title('Elbow Method For Optimal k')
# plt.show()

if __name__ == "__main__":
    main()