import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from preprocessing import preprocessing

data = pd.read_csv('heart_disease_patients.csv')
df = preprocessing(data)


# optimal numebr of clusters - elbow method
def num_of_clus_elbow(data):
    Sum_of_squared_distances = []
    K = range(1, 10)
    for num_clusters in K:
        kmeans = KMeans(n_clusters=num_clusters)
        kmeans.fit(data)
        Sum_of_squared_distances.append(kmeans.inertia_)

    # finding elbow point
    sd = np.diff(Sum_of_squared_distances, 2)
    elbow_point = np.argmax(sd) + 2
    return elbow_point


# # ploting inertia
# plt.plot(K,Sum_of_squared_distances,'bx-')
# plt.xlabel('Values of K') 
# plt.ylabel('Sum of squared distances') 
# plt.title('Elbow Method For Optimal k')
# plt.show()

# optimal number of clusters - silhouette method
def num_of_clus_silhouette(data):
    silhouette_scores = []
    for i in range(2, 11):
        kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=0)
        kmeans.fit(data)
        score = silhouette_score(data, kmeans.labels_)
        silhouette_scores.append(score)

    silhouette_point = np.argmax(silhouette_scores) + 2
    return silhouette_point


num_clusters = num_of_clus_silhouette(df)
kmeans = KMeans(n_clusters=num_clusters, init='k-means++', max_iter=300, n_init=10, random_state=0)
y_kmeans = kmeans.fit_predict(data)
