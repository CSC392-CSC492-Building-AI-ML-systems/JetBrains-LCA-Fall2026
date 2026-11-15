```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import MeanShift, MiniBatchKMeans, AgglomerativeClustering, SpectralClustering, DBSCAN, AffinityPropagation, Birch
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import kneighbors_graph
from sklearn.utils import shuffle
import warnings
import time

# Try to import HDBSCAN and OPTICS
try:
    from hdbscan import HDBSCAN
    hdbscan_available = True
except ImportError:
    hdbscan_available = False

try:
    from sklearn.cluster import OPTICS
    optics_available = True
except ImportError:
    optics_available = False

np.random.seed(0)

n_samples = 1500
noisy_circles = datasets.make_circles(n_samples=n_samples, factor=.5, noise=.05)
noisy_moons = datasets.make_moons(n_samples=n_samples, noise=.05)
blobs = datasets.make_blobs(n_samples=n_samples, random_state=8)
no_structure = (np.random.rand(n_samples, 2), None)

# Anisotropicly distributed data
X, y = datasets.make_blobs(n_samples=n_samples, random_state=170)
transformation = [[0.6, -0.6], [-0.4, 0.8]]
X_aniso = np.dot(X, transformation)
anisotropic = (X_aniso, y)

# Blobs with varied variances
X_varied, y_varied = datasets.make_blobs(
    n_samples=n_samples,
    cluster_std=[1.0, 2.5, 0.5],
    random_state=170)
varied = (X_varied, y_varied)

datasets_list = [
    ("Noisy Circles", noisy_circles),
    ("Noisy Moons", noisy_moons),
    ("Blobs", blobs),
    ("Anisotropic", anisotropic),
    ("Varied", varied),
    ("No Structure", no_structure)
]

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
    AgglomerativeClustering(n_clusters=3)
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

# HDBSCAN
if hdbscan_available:
    clustering_algorithms.append((
        "HDBSCAN",
        HDBSCAN(min_cluster_size=15)
    ))

# OPTICS
if optics_available:
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

plt.figure(figsize=(3 * len(clustering_algorithms), 18))

for i_dataset, (ds_name, dataset) in enumerate(datasets_list):
    X, y = dataset
    X = StandardScaler().fit_transform(X)
    for i_alg, (name, algorithm) in enumerate(clustering_algorithms):
        plt.subplot(len(datasets_list), len(clustering_algorithms), i_dataset * len(clustering_algorithms) + i_alg + 1)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning)
            warnings.filterwarnings("ignore", category=FutureWarning)
            t0 = time.time()
            try:
                if hasattr(algorithm, 'fit_predict'):
                    y_pred = algorithm.fit_predict(X)
                else:
                    y_pred = algorithm.fit(X).labels_
            except Exception as e:
                y_pred = np.full(X.shape[0], -1)
            t1 = time.time()
        colors = np.array([
            '#377eb8', '#ff7f00', '#4daf4a', '#f781bf', '#a65628',
            '#984ea3', '#999999', '#e41a1c', '#dede00'
        ])
        if len(np.unique(y_pred)) > len(colors):
            colors = plt.cm.tab20(np.linspace(0, 1, len(np.unique(y_pred))))
        plt.scatter(X[:, 0], X[:, 1], s=10, color=colors[y_pred % len(colors)])
        plt.title(f"{name}\n{t1 - t0:.2f}s", fontsize=9)
        plt.xticks([])
        plt.yticks([])
        if i_alg == 0:
            plt.ylabel(ds_name, fontsize=12)
plt.tight_layout()
plt.show()
```