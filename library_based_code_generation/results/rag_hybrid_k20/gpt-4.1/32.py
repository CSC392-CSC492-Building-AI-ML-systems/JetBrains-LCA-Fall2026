```python
import numpy as np
import matplotlib.pyplot as plt
import time
import warnings

from sklearn import cluster, mixture
from sklearn.neighbors import kneighbors_graph
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_circles, make_moons, make_blobs, make_classification

# HDBSCAN and OPTICS
try:
    import hdbscan
    has_hdbscan = True
except ImportError:
    has_hdbscan = False

from sklearn.cluster import (
    MeanShift, MiniBatchKMeans, AgglomerativeClustering, SpectralClustering,
    DBSCAN, OPTICS, AffinityPropagation, Birch
)
from sklearn.mixture import GaussianMixture

np.random.seed(0)

# Generate datasets
n_samples = 1500
datasets = [
    (
        make_circles(n_samples=n_samples, factor=0.5, noise=0.05),
        "Noisy Circles"
    ),
    (
        make_moons(n_samples=n_samples, noise=0.05),
        "Noisy Moons"
    ),
    (
        make_blobs(n_samples=n_samples, random_state=8),
        "Blobs"
    ),
    (
        (np.dot(make_blobs(n_samples=n_samples, random_state=170)[0],
                [[0.6, -0.6], [-0.4, 0.8]]), make_blobs(n_samples=n_samples, random_state=170)[1]),
        "Anisotropic Blobs"
    ),
    (
        make_blobs(n_samples=n_samples, cluster_std=[1.0, 2.5, 0.5], random_state=170),
        "Varied Blobs"
    ),
    (
        (np.random.rand(n_samples, 2), None),
        "No Structure"
    )
]

# Set up clustering algorithms
clustering_algorithms = []

bandwidth = cluster.estimate_bandwidth(datasets[2][0][0], quantile=0.3)
connectivity = kneighbors_graph(datasets[2][0][0], n_neighbors=10, include_self=False)
connectivity = 0.5 * (connectivity + connectivity.T)

clustering_algorithms.append((
    "MiniBatchKMeans",
    MiniBatchKMeans(n_clusters=3, n_init=10)
))
clustering_algorithms.append((
    "MeanShift",
    MeanShift(bandwidth=bandwidth, bin_seeding=True)
))
clustering_algorithms.append((
    "AgglomerativeClustering",
    AgglomerativeClustering(n_clusters=3, linkage='ward', connectivity=connectivity)
))
clustering_algorithms.append((
    "SpectralClustering",
    SpectralClustering(n_clusters=3, eigen_solver='arpack', affinity="nearest_neighbors", assign_labels='kmeans')
))
clustering_algorithms.append((
    "DBSCAN",
    DBSCAN(eps=0.3)
))
if has_hdbscan:
    clustering_algorithms.append((
        "HDBSCAN",
        hdbscan.HDBSCAN(min_cluster_size=15)
    ))
clustering_algorithms.append((
    "OPTICS",
    OPTICS(min_samples=10)
))
clustering_algorithms.append((
    "AffinityPropagation",
    AffinityPropagation(damping=0.9, preference=-200)
))
clustering_algorithms.append((
    "Birch",
    Birch(n_clusters=3)
))
clustering_algorithms.append((
    "GaussianMixture",
    GaussianMixture(n_components=3, covariance_type='full')
))

plt.figure(figsize=(len(clustering_algorithms) * 3, len(datasets) * 3))
plot_num = 1

for i_dataset, (dataset, dataset_name) in enumerate(datasets):
    X, y = dataset
    X = StandardScaler().fit_transform(X)

    for name, algorithm in clustering_algorithms:
        t0 = time.time()
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="the number of connected components of the connectivity matrix is [0-9]+"
            )
            warnings.filterwarnings(
                "ignore",
                message="Graph is not fully connected, spectral embedding may not work as expected."
            )
            try:
                if name == "GaussianMixture":
                    algorithm.fit(X)
                    y_pred = algorithm.predict(X)
                else:
                    algorithm.fit(X)
                    if hasattr(algorithm, 'labels_'):
                        y_pred = algorithm.labels_.astype(int)
                    else:
                        y_pred = algorithm.predict(X)
            except Exception:
                y_pred = np.full(X.shape[0], -1)
        t1 = time.time()
        plt.subplot(len(datasets), len(clustering_algorithms), plot_num)
        if y_pred is not None:
            colors = np.array(
                [plt.cm.nipy_spectral(float(i) / max(y_pred.max() + 1, 1))
                 if i >= 0 else (0, 0, 0, 1) for i in y_pred]
            )
        else:
            colors = 'k'
        plt.scatter(X[:, 0], X[:, 1], s=10, color=colors)
        plt.title(f"{name}\n{dataset_name}\n{t1 - t0:.2f}s", fontsize=9)
        plt.xticks([])
        plt.yticks([])
        plot_num += 1

plt.tight_layout()
plt.show()
```