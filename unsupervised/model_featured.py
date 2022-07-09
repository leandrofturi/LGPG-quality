# https://scikit-learn.org/stable/modules/clustering.html
import time

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler


np.random.seed(42)


def learn(X, K, out_filename):
    dataset = out_filename.split("/")[1].split(".")[0]
    print(f"Starting {dataset}...")
    start = time.time()

    kmeans = KMeans(n_clusters=K)
    kmeans.fit(X)
    y = kmeans.predict(X)

    X_embedded = TSNE(n_components=2, init="pca").fit_transform(X)

    cmap = matplotlib.cm.get_cmap("viridis", K)
    fig, ax = plt.subplots()
    for g in range(kmeans.n_clusters):
        ix = np.where(y == g)
        ax.scatter(
            X_embedded[ix, 0],
            X_embedded[ix, 1],
            label=g,
            c=matplotlib.colors.rgb2hex(cmap(g)),
        )
    plt.xlabel('Primeira componente')
    plt.ylabel('Segunda componente')
    plt.grid(False)
    ax.legend(frameon=True, framealpha=1, edgecolor="black")
    plt.tight_layout()
    plt.savefig(f"output/scatter_{dataset}.png")

    # feature importance
    # https://github.com/YousefGh/kmeans-feature-importance
    labels = kmeans.n_clusters
    ordered_feature_names = X.columns
    centroids = kmeans.cluster_centers_
    centroids = np.vectorize(lambda x: np.abs(x))(centroids)
    sorted_centroid_features_idx = centroids.argsort(axis=1)[:, ::-1]

    cluster_feature_weights = {}
    for label, centroid in zip(range(labels), sorted_centroid_features_idx):
        ordered_cluster_feature_weights = centroids[label][
            sorted_centroid_features_idx[label]
        ]
        ordered_cluster_feature_weights = ordered_cluster_feature_weights / sum(
            ordered_cluster_feature_weights
        )
        ordered_cluster_features = [
            ordered_feature_names[feature] for feature in centroid
        ]
        cluster_feature_weights[label] = pd.DataFrame(
            zip(ordered_cluster_features, ordered_cluster_feature_weights),
            columns=["variable", "weight"],
        )

    fig, ax = plt.subplots()
    cm = plt.get_cmap("viridis")
    ax.set_prop_cycle(color=[cm(1.0 * i / labels) for i in range(labels)])
    for k, v in cluster_feature_weights.items():
        ax.bar(v["variable"][:5], v["weight"][:5], label=k)
    ax.legend()
    plt.xticks(rotation=45)
    plt.title(f"{dataset} k={K}")
    plt.tight_layout()
    plt.savefig(f"output/feature_weights_{dataset}.png")

    print("ellapsed time is {:.6f} seconds".format(time.time() - start))

    return y
