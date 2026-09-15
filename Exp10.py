from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.datasets import load_iris
import pandas as pd
import matplotlib.pyplot as plt

# Load Iris dataset
data = load_iris()
X = pd.DataFrame(data.data)

# K-Means
kmeans = KMeans(n_clusters=3, random_state=0)
kmeans.fit(X)
k_pred = kmeans.labels_

# GMM
gmm = GaussianMixture(n_components=3, random_state=0)
gmm.fit(X)
g_pred = gmm.predict(X)

# Plot
plt.figure(figsize=(12,4))

plt.subplot(1,3,1)
plt.scatter(X.iloc[:,2], X.iloc[:,3], c=data.target)
plt.title("Real")

plt.subplot(1,3,2)
plt.scatter(X.iloc[:,2], X.iloc[:,3], c=k_pred)
plt.title("K-Means")

plt.subplot(1,3,3)
plt.scatter(X.iloc[:,2], X.iloc[:,3], c=g_pred)
plt.title("GMM")

plt.show()
