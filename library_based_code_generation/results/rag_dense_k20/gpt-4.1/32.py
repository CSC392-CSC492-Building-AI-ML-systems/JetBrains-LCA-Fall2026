```python
import numpy as np
import matplotlib.pyplot as plt
import time
import warnings

from sklearn import cluster, mixture
from sklearn.neighbors import kneighbors_graph
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_circles, make_moons, make_blobs, make_classification

# Try to import HDBSCAN
try:
    import hdbscan
    has_hdbscan = True
except ImportError:
    has_hdbscan = False

# Generate datasets
n_samples = 1500
random_state = 170

datasets = [
    (
        make_circles(n_samples=n_samples, factor=0.5, noise=0.05),
        {"name": "Noisy Circles"}
    ),
    (
        make_moons(n_samples=n_samples, noise=0.05),
        {"name": "Noisy Moons"}
    ),
    (
        make_blobs(n_samples=n_samples, random_state=8),
        {"name": "Blobs"}
    ),
    (
        # Anisotropicly distributed data
        (np.dot(make_blobs(n_samples=n_samples, random_state=random_state)[0],
                [[0.6, -0.6], [-0.4, 0.8]]),
         make_blobs(n_samples=n_samples, random_state=random_state)[1]),
        {"name": "Anisotropic Blobs"}
    ),
    (
        # Blobs with varied variances
        make_blobs(n_samples=n_samples,
                   cluster_std=[1.0, 2.5, 0.5],
                   random_state=random_state),
        {"name": "Varied Blobs"}
    ),
    (
        # No structure
        (np.random.rand(n_samples, 2), None),
        {"name": "No Structure"}
    ),
]

# Set up clustering algorithms
default_base = {
    "quantile": .3,
    "eps": .3,
    "damping": .9,
    "preference": -200,
    "n_neighbors": 10,
    "n_clusters": 3,
    "min_samples": 20,
    "xi": 0.05,
    "min_cluster_size": 0.1,
}

clustering_algorithms = []

# MeanShift
clustering_algorithms.append((
    "MeanShift",
    cluster.MeanShift(bin_seeding=True)
))

# MiniBatchKMeans
clustering_algorithms.append((
    "MiniBatchKMeans",
    cluster.MiniBatchKMeans(n_clusters=default_base["n_clusters"])
))

# AgglomerativeClustering
clustering_algorithms.append((
    "AgglomerativeClustering",
    cluster.AgglomerativeClustering(n_clusters=default_base["n_clusters"])
))

# SpectralClustering
clustering_algorithms.append((
    "SpectralClustering",
    cluster.SpectralClustering(
        n_clusters=default_base["n_clusters"],
        eigen_solver='arpack',
        affinity="nearest_neighbors",
        assign_labels="kmeans"
    )
))

# DBSCAN
clustering_algorithms.append((
    "DBSCAN",
    cluster.DBSCAN(eps=default_base["eps"])
))

# HDBSCAN (if available)
if has_hdbscan:
    clustering_algorithms.append((
        "HDBSCAN",
        hdbscan.HDBSCAN(min_cluster_size=int(default_base["min_cluster_size"] * n_samples))
    ))

# OPTICS
clustering_algorithms.append((
    "OPTICS",
    cluster.OPTICS(min_samples=default_base["min_samples"], xi=default_base["xi"])
))

# AffinityPropagation
clustering_algorithms.append((
    "AffinityPropagation",
    cluster.AffinityPropagation(damping=default_base["damping"], preference=default_base["preference"], random_state=0)
))

# Birch
clustering_algorithms.append((
    "Birch",
    cluster.Birch(n_clusters=default_base["n_clusters"])
))

# GaussianMixture
clustering_algorithms.append((
    "GaussianMixture",
    mixture.GaussianMixture(n_components=default_base["n_clusters"], covariance_type='full', random_state=0)
))

plt.figure(figsize=(len(clustering_algorithms) * 3, len(datasets) * 3))
plt.subplots_adjust(left=.02, right=.98, bottom=.001, top=.96, wspace=.05, hspace=.01)

for i_dataset, (dataset, ds_info) in enumerate(datasets):
    X, y = dataset
    X = StandardScaler().fit_transform(X)

    # Compute connectivity matrix for structured Ward
    connectivity = kneighbors_graph(X, n_neighbors=default_base["n_neighbors"], include_self=False)
    # Make connectivity symmetric
    connectivity = 0.5 * (connectivity + connectivity.T)

    for i_alg, (name, algorithm) in enumerate(clustering_algorithms):
        t0 = time.time()
        # Some algorithms need special handling
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning)
            warnings.filterwarnings("ignore", category=FutureWarning)
            if name == "AgglomerativeClustering":
                algorithm.set_params(connectivity=connectivity)
                y_pred = algorithm.fit_predict(X)
            elif name == "SpectralClustering":
                try:
                    y_pred = algorithm.fit_predict(X)
                except Exception:
                    y_pred = np.zeros(X.shape[0], dtype=int)
            elif name == "GaussianMixture":
                y_pred = algorithm.fit(X).predict(X)
            else:
                try:
                    y_pred = algorithm.fit_predict(X)
                except Exception:
                    y_pred = np.zeros(X.shape[0], dtype=int)
        t1 = time.time()
        plt.subplot(len(datasets), len(clustering_algorithms), i_dataset * len(clustering_algorithms) + i_alg + 1)
        colors = np.array(
            [plt.cm.nipy_spectral(float(i) / max(y_pred.max() + 1, 1)) for i in y_pred]
        )
        plt.scatter(X[:, 0], X[:, 1], s=10, color=colors)
        plt.xticks([])
        plt.yticks([])
        if i_dataset == 0:
            plt.title(name, size=10)
        if i_alg == 0:
            plt.ylabel(ds_info["name"], size=10)
        plt.text(.99, .01, ('%.2fs' % (t1 - t0)).lstrip('0'),
                 transform=plt.gca().transAxes, size=9,
                 horizontalalignment='right')
plt.tight_layout()
plt.show()
```