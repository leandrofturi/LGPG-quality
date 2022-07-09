# https://scikit-learn.org/stable/modules/clustering.html
import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler


np.random.seed(42)


def learn(df, K, out_filename):
    dataset = out_filename.split('/')[1].split('.')[0]
    print(f"Starting {dataset}...")
    start = time.time()
    
    df_sample = df.iloc[: min(len(df.index), int(1e5)), :]
    print(df_sample.shape)

    enc = {}
    for c in df_sample.select_dtypes(include=["string", "object", "category"]).columns:
        enc[c] = LabelEncoder()
        df_sample.loc[df_sample.index, c] = enc[c].fit_transform(
            df_sample[c].astype(str)
        )

    df_sample.is_copy = False
    df_sample.fillna(-1, inplace=True)
    X = StandardScaler().fit_transform(df_sample)

    kmeans = KMeans(n_clusters=K)
    kmeans.fit(X)
    y = kmeans.predict(X)

    X_embedded = TSNE(n_components=2, unsupervised_rate="auto", init="pca").fit_transform(X)

    plt.figure()
    scatter = plt.scatter(X_embedded[:, 0], X_embedded[:, 1], c=y, cmap="viridis")
    plt.legend(handles=scatter.legend_elements()[0], labels=range(kmeans.n_clusters))
    plt.title(f'{dataset} k={K}')
    plt.tight_layout()
    plt.savefig(f"output/scatter_{dataset}.png")

    # feature importance
    # https://github.com/YousefGh/kmeans-feature-importance
    labels = kmeans.n_clusters
    ordered_feature_names = df.columns
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
    plt.title(f'{dataset} k={K}')
    plt.tight_layout()
    plt.savefig(f"output/feature_weights_{dataset}.png")

    print("ellapsed time is {:.6f} seconds".format(time.time() - start))
