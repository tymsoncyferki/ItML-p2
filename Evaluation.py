from sklearn.metrics import silhouette_score, calinski_harabasz_score, silhouette_samples
import matplotlib.pyplot as plt
from preprocessing import preprocessing
from KMeans import k_means_alg, k_prototypes_alg
from OtherAlgorithms import dbscan
import numpy as np
import pandas as pd


def medians_for_clusters(df):
    medians = df.groupby('label').median()
    return medians


def plot_all_columns(df):
    label_column = 'label'

    # Select all numeric columns for scatter plots
    numeric_columns = df.select_dtypes(include='number').columns

    # Define the number of rows and columns in the subplot grid
    num_rows = num_cols = len(numeric_columns)

    # Create a new figure and subplot grid
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(12, 12))

    # Iterate through each pair of columns
    for i, x_column in enumerate(numeric_columns):
        for j, y_column in enumerate(numeric_columns):
            if x_column == label_column or y_column == label_column:
                continue
            if x_column != y_column:
                fig, ax = plt.subplots(figsize=(6, 6))
                ax.scatter(df[x_column], df[y_column], c=df[label_column], cmap='viridis')
                ax.set_xlabel(x_column)
                ax.set_ylabel(y_column)
                ax.set_title(f'{x_column} vs {y_column}')

                filename = f'plots/scatter_plot_{x_column}_vs_{y_column}.png'
                plt.savefig(filename)

                plt.close()


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

    # (k_prototypes_alg, 'K Prototypes')
    # algorithms = [(k_means_alg, 'K Means')]
    # for alg, name in algorithms:
    #     print('-----')
    #     print(name)
    #     df = preprocessing(data, rem_over=True, process='minmax')
    #     df_with_labels, model = alg(df)
    #     df_without_labels = df_with_labels.drop("label", axis=1)
    #     y = df_with_labels['label']
    #     print(f"silhouette score: {silhouette_score(df_without_labels, y)}")
    #     print(f"calinski harabasz score: {calinski_harabasz_score(df_without_labels, y)}")
    #     data_with_labels = data.copy()
    #     data_with_labels['label'] = df_with_labels['label']
    #     print(medians_for_clusters(data_with_labels))
    #     plot_all_columns(df_with_labels)


if __name__ == "__main__":
    main()

# def plot_silhouette(df, y_kmeans):
#     """not important"""
#     # Compute silhouette scores for each sample
#     silhouette_avg = silhouette_score(df, y_kmeans)
#     sample_silhouette_values = silhouette_samples(df, y_kmeans)
#     n = len(np.unique(y_kmeans))
#     # Plot silhouette plot for 2 clusters
#     fig, ax = plt.subplots()
#     y_lower = 10
#     for i in range(n):
#         # Aggregate the silhouette scores for samples belonging to cluster i
#         ith_cluster_silhouette_values = \
#             sample_silhouette_values[y_kmeans == i]
#         ith_cluster_silhouette_values.sort()
#         size_cluster_i = ith_cluster_silhouette_values.shape[0]
#         y_upper = y_lower + size_cluster_i
#
#         color = plt.cm.Spectral(float(i) / n)
#         ax.fill_betweenx(np.arange(y_lower, y_upper),
#                          0, ith_cluster_silhouette_values,
#                          facecolor=color, edgecolor=color, alpha=0.7)
#
#         # Label silhouette plots with their cluster numbers at the middle
#         ax.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))
#
#         # Compute the new y_lower for next plot
#         y_lower = y_upper + 10
#
#     ax.set_title(f"Silhouette plot for {n} clusters")
#     ax.set_xlabel("Silhouette coefficient values")
#     ax.set_ylabel("Cluster labels")
#
#     # Add vertical line for average silhouette score
#     ax.axvline(x=silhouette_avg, color="red", linestyle="--")
#
#     # Set the x-axis limits and ticks
#     ax.set_xlim([-0.1, 1])
#     ax.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])
#
#     # Set the y-axis limits and ticks
#     ax.set_ylim([0, y_upper + 10])
#     ax.set_yticks([])
#
#     plt.show()


# # ploting inertia
# plt.plot(K,Sum_of_squared_distances,'bx-')
# plt.xlabel('Values of K')
# plt.ylabel('Sum of squared distances')
# plt.title('Elbow Method For Optimal k')
# plt.show()
