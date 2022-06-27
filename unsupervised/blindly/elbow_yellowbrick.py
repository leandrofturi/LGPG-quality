# https://scikit-learn.org/stable/modules/clustering.html
import time
from os import listdir
from os.path import isfile, join

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from yellowbrick.cluster.elbow import kelbow_visualizer


np.random.seed(42)

files = [f for f in listdir("datasets") if isfile(join("datasets", f))]

for file in files:
    print(file)
    start = time.time()

    df = pd.read_parquet(f"datasets\\{file}")
    df = df.iloc[: min(len(df.index), int(1e5)), :]
    print(df.shape)

    enc = {}
    for c in df.select_dtypes(include=["string", "object", "category"]).columns:
        enc[c] = LabelEncoder()
        df[c] = enc[c].fit_transform(df[c].astype(str))

    df.fillna(-1, inplace=True)
    X = StandardScaler().fit_transform(df)

    # squaredDistances = []
    # for k in range(2, 24):
    # kmeans = KMeans(n_clusters=k)
    # kmeans.fit(X)
    # squaredDistances.append(kmeans.inertia_)
    # plt.plot(range(2, 24), squaredDistances, "bx-")
    # plt.axvline(x=10, color="r", linestyle="dashed")
    # plt.show()
    # plt.title(file.replace(".parquet", ""))

    kelbow_visualizer(KMeans(), X, k=(2, 24), title=file.replace(".parquet", ""))

    print("ellapsed time is {:.6f} seconds".format(time.time() - start))
