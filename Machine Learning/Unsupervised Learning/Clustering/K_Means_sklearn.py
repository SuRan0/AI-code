from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt

# generate some sample data
# random_state=0: same data -> useful for debug and experiment
# y is label, although this is unsupervised learning, we can use y to analyse better.
X, y = make_blobs(n_samples=300, centers=4, random_state=0, cluster_std=0.6)

# fit the k-means clustering model
km = KMeans(n_clusters=4)
km.fit(X)

# predict the cluster for each data point
y_pred = km.predict(X)

# plot the samples
plt.scatter(X[:, 0], X[:, 1], c=y, cmap='rainbow')
plt.show()
