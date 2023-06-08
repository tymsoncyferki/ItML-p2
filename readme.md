# Heart disease clusterization
This project was created for Machine Learning course at the Warsaw University of Technology. Our task was to cluster anonymized data of patients who have been diagnosed with heart disease to help doctors understand which treatments might work with their patients as patients with similar characteristics may respond to the same treatments.

## Authors
[Tymoteusz Urban](https://github.com/tymsoncyferki)<br>
[Karolina Mączka](https://github.com/KarolinaMaczka)

## Data
[Heart Disease patients](https://www.kaggle.com/datasets/kingabzpro/heart-disease-patients)

## Data exploration
We did a thorough study of the data in order to fully understand the dataset we have been working with. We also asked medical expert to give us medical insights and help us with more specific interpretation of every feature. Data was well prepared, all columns were numerical and there were no null values. 

## Preprocessing
At first we wanted to delete outliers but after consultation with our validation team we abandoned this idea as outliers are important in medical data analysis. After many tests we decided to apply MinMax scaler and reduce dimensionality with PCA. 

## Model
We obtained optimal number of clusters from Silhouette method. After testing multiple clustering algorithms we have chosen KMeans - the most common algorithm yet the most effective. It got the best results in almost all metrics. We also checked the clustering on 3D visualizations of principal components:

<img src="https://github.com/tymsoncyferki/heart-disease-clusterization/blob/main/data/pca_visualization.png" width=500>

## Interpretation
We created random forest model in order to extract feature importance for each cluster. By analyzing means, medians, boxplot charts and feature importance we could create descriptions of each cluster. To learn more about the results and whole process of clusterization open project_presentation file. 
