```python
import numpy as np
import matplotlib.pyplot as plt
import time
import warnings

from sklearn import cluster, mixture
from sklearn.neighbors import kneighbors_graph
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_circles, make_moons, make_blobs, make_classification

# Try to import HDBSCAN, skip if not available
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
        # Anisotropicly distributed data
        (np.dot(make_blobs(n_samples=n_samples, random_state=random_state)[0], [[0.6, -0.6], [-0.4, 0.8]]),
         make_blobs(n_samples=n_samples, random_state=random_state)[1]),
        "Anisotropic Blobs"
    ),
    (
        # Blobs with varied variances
        make_blobs(n_samples=n_samples,
                   cluster_std=[1.0, 2.5, 0.5],
                   random_state=random_state),
        "Varied Blobs"
    ),
    (
        # No structure
        (np.random.rand(n_samples, 2), None),
        "No Structure"
    ),
]

# Set up clustering algorithms
clustering_algorithms = []

# MeanShift
bandwidth = cluster.estimate_bandwidth(datasets[2][0][0], quantile=0.3)
clustering_algorithms.append((
    "MeanShift",
    cluster.MeanShift(bandwidth=bandwidth, bin_seeding=True)
))

# MiniBatchKMeans
clustering_algorithms.append((
    "MiniBatchKMeans",
    cluster.MiniBatchKMeans(n_clusters=3)
))

# AgglomerativeClustering
clustering_algorithms.append((
    "AgglomerativeClustering",
    cluster.AgglomerativeClustering(n_clusters=3)
))

# SpectralClustering
clustering_algorithms.append((
    "SpectralClustering",
    cluster.SpectralClustering(n_clusters=3, affinity="nearest_neighbors", assign_labels="kmeans")
))

# DBSCAN
clustering_algorithms.append((
    "DBSCAN",
    cluster.DBSCAN(eps=0.3)
))

# HDBSCAN (if available)
if has_hdbscan:
    clustering_algorithms.append((
        "HDBSCAN",
        hdbscan.HDBSCAN(min_cluster_size=15)
    ))

# OPTICS
clustering_algorithms.append((
    "OPTICS",
    cluster.OPTICS(min_samples=10)
))

# AffinityPropagation
clustering_algorithms.append((
    "AffinityPropagation",
    cluster.AffinityPropagation(damping=0.9, preference=-200)
))

# Birch
clustering_algorithms.append((
    "Birch",
    cluster.Birch(n_clusters=3)
))

# GaussianMixture
clustering_algorithms.append((
    "GaussianMixture",
    mixture.GaussianMixture(n_components=3, covariance_type='full')
))

plt.figure(figsize=(len(clustering_algorithms) * 3 + 3, len(datasets) * 3))
plt.subplots_adjust(left=.02, right=.98, bottom=.001, top=.96, wspace=.05, hspace=.01)

for i_dataset, (dataset, dataset_name) in enumerate(datasets):
    X, y = dataset
    X = StandardScaler().fit_transform(X)

    # Compute connectivity matrix for structured Ward
    connectivity = None
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="kneighbors_graph")
        try:
            connectivity = kneighbors_graph(X, n_neighbors=10, include_self=False)
            # make connectivity symmetric
            connectivity = 0.5 * (connectivity + connectivity.T)
        except Exception:
            connectivity = None

    for i_alg, (name, algorithm) in enumerate(clustering_algorithms):
        t0 = time.time()
        params = algorithm.get_params() if hasattr(algorithm, "get_params") else {}
        # Set connectivity if applicable
        if name == "AgglomerativeClustering" and connectivity is not None:
            algorithm.set_params(connectivity=connectivity)
        if name == "SpectralClustering":
            algorithm.set_params(n_init=10)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning)
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
            plt.ylabel(dataset_name, size=10)
        plt.text(.99, .01, ("%.2fs" % (t1 - t0)).lstrip("0"),
                 transform=plt.gca().transAxes, size=9,
                 horizontalalignment='right')

plt.tight_layout()
plt.show()
```