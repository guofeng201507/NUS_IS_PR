import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

#(1) Read in your dataset
data = pd.read_excel('BirthDeathRates.xlsx',sheet_name = "Sheet1")

# print(data.head())

#(2) Pre-processing & cleaning
work_data = data.copy()
# plt.hist(work_data)
# plt.show()

work_data = work_data.drop(['SN', 'Country'], axis=1)
print(work_data.describe())

#(3) Standardize data
X_std = StandardScaler().fit_transform(work_data)

for n_clusters in range(2, 10):
    clusterer = KMeans (n_clusters=n_clusters)
    preds = clusterer.fit_predict(X_std)

    plt.scatter(X_std[:, 2], X_std[:, 3], c=preds, s=50, cmap='viridis')
    plt.title(n_clusters)
    plt.show()

    centers = clusterer.cluster_centers_

    score = silhouette_score (X_std, preds, metric='euclidean')
    print ("For n_clusters = {}, silhouette score is {})".format(n_clusters, score))



