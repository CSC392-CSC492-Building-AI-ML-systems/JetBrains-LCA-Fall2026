```python
import numpy as np
import matplotlib.pyplot as plt
import time
import warnings

from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import (
    MeanShift, MiniBatchKMeans, AgglomerativeClustering, SpectralClustering,
    DBSCAN, AffinityPropagation, Birch
)
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import kneighbors_graph

# HDBSCAN and OPTICS
try:
    import hdbscan
except ImportError:
    hdbscan = None
from sklearn.cluster import OPTICS

# Ignore warnings from kneighbors_graph
warnings.filterwarnings("ignore", category=UserWarning)

# Generate datasets
n_samples = 1500
random_state = 170

datasets_list = [
    (
        datasets.make_circles(n_samples=n_samples, factor=0.5, noise=0.05),
        "Noisy Circles"
    ),
    (
        datasets.make_moons(n_samples=n_samples, noise=0.05),
        "Noisy Moons"
    ),
    (
        datasets.make_blobs(n_samples=n_samples, random_state=8),
        "Blobs"
    ),
    (
        (np.dot(datasets.make_blobs(n_samples=n_samples, random_state=random_state)[0],
                [[0.6, -0.6], [-0.4, 0.8]]),
         datasets.make_blobs(n_samples=n_samples, random_state=random_state)[1]),
        "Anisotropic Blobs"
    ),
    (
        datasets.make_blobs(n_samples=n_samples,
                            cluster_std=[1.0, 2.5, 0.5],
                            random_state=random_state),
        "Varied Blobs"
    ),
    (
        (np.random.rand(n_samples, 2), None),
        "No Structure"
    )
]

# Set up clustering algorithms
clustering_algorithms = []

# MeanShift
clustering_algorithms.append((
    "MeanShift",
    MeanShift(bin_seeding=True)
))

# MiniBatchKMeans
clustering_algorithms.append((
    "MiniBatchKMeans",
    MiniBatchKMeans(n_clusters=3)
))

# AgglomerativeClustering
clustering_algorithms.append((
    "AgglomerativeClustering",
    AgglomerativeClustering(n_clusters=3, linkage='ward')
))

# SpectralClustering
clustering_algorithms.append((
    "SpectralClustering",
    SpectralClustering(n_clusters=3, affinity='nearest_neighbors', assign_labels='kmeans')
))

# DBSCAN
clustering_algorithms.append((
    "DBSCAN",
    DBSCAN(eps=0.3)
))

# HDBSCAN (if available)
if hdbscan is not None:
    clustering_algorithms.append((
        "HDBSCAN",
        hdbscan.HDBSCAN(min_cluster_size=15)
    ))

# OPTICS
clustering_algorithms.append((
    "OPTICS",
    OPTICS(min_samples=10)
))

# AffinityPropagation
clustering_algorithms.append((
    "AffinityPropagation",
    AffinityPropagation(damping=0.9, preference=-200)
))

# Birch
clustering_algorithms.append((
    "Birch",
    Birch(n_clusters=3)
))

# GaussianMixture
class GMMCluster:
    def __init__(self, n_components=3):
        self.model = GaussianMixture(n_components=n_components)
    def fit_predict(self, X):
        self.model.fit(X)
        return self.model.predict(X)
clustering_algorithms.append((
    "GaussianMixture",
    GMMCluster(n_components=3)
))

plt.figure(figsize=(len(clustering_algorithms) * 3, len(datasets_list) * 3))
plot_num = 1

for i_dataset, (dataset, dataset_name) in enumerate(datasets_list):
    X, y = dataset
    X = StandardScaler().fit_transform(X)

    # Compute connectivity matrix for structured Ward
    connectivity = kneighbors_graph(X, n_neighbors=10, include_self=False)
    connectivity = 0.5 * (connectivity + connectivity.T)

    for i_alg, (name, algorithm) in enumerate(clustering_algorithms):
        t0 = time.time()
        # Some algorithms need special handling
        if name == "AgglomerativeClustering":
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                algorithm.set_params(connectivity=connectivity)
                y_pred = algorithm.fit_predict(X)
        elif name == "SpectralClustering":
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                y_pred = algorithm.fit_predict(X)
        elif name == "GaussianMixture":
            y_pred = algorithm.fit_predict(X)
        else:
            y_pred = algorithm.fit_predict(X)
        t1 = time.time()
        plt.subplot(len(datasets_list), len(clustering_algorithms), plot_num)
        if y_pred is not None:
            plt.scatter(X[:, 0], X[:, 1], c=y_pred, s=10, cmap='viridis')
        else:
            plt.scatter(X[:, 0], X[:, 1], s=10)
        plt.title(f"{name}\n{t1 - t0:.2f}s", fontsize=9)
        plt.xticks([])
        plt.yticks([])
        if i_alg == 0:
            plt.ylabel(dataset_name, fontsize=10)
        plot_num += 1

plt.tight_layout()
plt.show()
```