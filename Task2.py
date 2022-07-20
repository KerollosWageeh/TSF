# KEROLLOS WAGEEH

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.cluster import KMeans

# Reading data from iris dataset
iris = datasets.load_iris()
iris_data = pd.DataFrame(iris.data, columns= iris.feature_names)

# Using the elbow method to predict the optimum number of clusters
# Get the value of "within cluster sum of squares" (wcss) for range of numbers 1-10
# Plot wcss against the different number of clusters 1-10 to apply the elbow method
x = iris_data.iloc[:, :].values
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters = i)
    kmeans.fit(x)
    wcss.append(kmeans.inertia_)

plt.plot(range(1,11),wcss)
plt.xlabel('# of clusters')
plt.ylabel('Within cluster sum of squares')
plt.show()


# Optimal number of clusters is 3
# Use 3 to predict the labels and scatter them to represent them visually as clusters
kmeans = KMeans(n_clusters = 3)
kmeans.fit(x)
plt.scatter(x[kmeans.labels_ == 0, 0], x[kmeans.labels_ == 0, 1], 
            s = 100, c = 'red', label = 'Iris-setosa')
plt.scatter(x[kmeans.labels_ == 1, 0], x[kmeans.labels_ == 1, 1], 
            s = 100, c = 'blue', label = 'Iris-versicolour')
plt.scatter(x[kmeans.labels_ == 2, 0], x[kmeans.labels_ == 2, 1],
            s = 100, c = 'green', label = 'Iris-virginica')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:,1], 
            s = 100, c = 'yellow', label = 'Centroids')


plt.legend()
plt.show()