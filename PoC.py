import os.path
import timeit
from os import listdir
from os.path import isfile
from pathlib import Path
from typing import List, Optional

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import sklearn.utils.validation
from scipy.spatial.distance import pdist
from scipy.cluster.hierarchy import (
    linkage,
    cophenet,
    fcluster,
)
from sklearn.decomposition import PCA
from sklearn.metrics import (
    silhouette_score,
    calinski_harabasz_score,
    davies_bouldin_score,
)
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier, NearestNeighbors
from sklearn.preprocessing import OrdinalEncoder, StandardScaler

from utils import Dataset, DataType, Data
from GowerMetric import GowerMetric
from GowerMetricTestClass import GowerMetric2


def bin_dist(vector_1: np.ndarray, vector_2: np.ndarray):
    # bool array for not-null values
    non_null_map_1 = vector_1 != -1
    non_null_map_2 = vector_2 != -1
    non_null_map = non_null_map_1 & non_null_map_2

    # vec_1 and vec_2 with only not-null values
    non_null_1 = vector_1[non_null_map]
    non_null_2 = vector_2[non_null_map]

    # count null values
    null_count = (~non_null_map_1 | ~non_null_map_2).sum()

    # return sum of dissimilarities between vec_1 and vec_2 + number of unique null fields
    return ((~np.isclose(non_null_1, non_null_2)).sum() + null_count) / len(
        vector_1
    )


# Simple function for calculating dissimilarity matrix
def make_matrix(data_frame: np.ndarray, metric) -> np.array:
    return pdist(data_frame, metric)


def cpcc(X, Z):
    return cophenet(Z, X)


def ioa(O, P):
    O_ = np.average(O)
    return 1 - np.sum(np.power(P - O, 2)) / np.sum(
        np.power(
            np.absolute(P - O_) + np.absolute(O - O_),
            2,
        )
    )


def silhouette_test(Z, df, metric_func):
    scores = []

    for i in range(3, 10):
        pred_labels = fcluster(Z, t=i, criterion="maxclust")
        if len(np.unique(pred_labels)) > 1:
            scores.append(
                silhouette_score(df, pred_labels, metric=metric_func)
            )
        else:
            scores.append(0.0)

    scores = np.array(scores)
    X_values = np.linspace(3, 10, 7)
    plt.plot(X_values, scores)
    plt.show()


def pca_test(df: np.ndarray, y: np.ndarray = None, labels=None):
    if y is None and labels is None:
        print(
            "PCA Test Error - data is not labeled neither by y nor by predicted labels!"
        )
        return

    colors = list(mcolors.CSS4_COLORS.values())
    df_cp = df.copy()

    df_cp = StandardScaler().fit_transform(df_cp)

    pca = PCA(n_components=2)
    principal_components = pca.fit_transform(df_cp)

    if labels is not None:
        y = labels
        colors = colors[: np.max(labels)]
    else:
        colors = colors[: np.max(y)]

    y = y.copy()
    y = y.reshape(-1, 1)

    principal_components = np.concatenate((principal_components, y), axis=1)

    for e, color in enumerate(colors):
        plt.scatter(
            x=[
                principal_component[0]
                for principal_component in principal_components[
                    principal_components[:, 2] == e + 1
                ]
            ],
            y=[
                principal_component[1]
                for principal_component in principal_components[
                    principal_components[:, 2] == e + 1
                ]
            ],
            c=color,
            label=str(e + 1),
        )

    print(np.bincount(labels))
    plt.legend()
    plt.show()


def mertic_test(
    gower,
    dataset: Dataset,
    data: Data,
    number_of_records: int = None,
):
    if number_of_records is None:
        number_of_records = len(data.data[dataset.name])

    df = np.copy(data.data[dataset.name][:number_of_records])
    df = fill_na(df)

    # gower = GowerMetric(
    #     data.cols_type[dataset.name],
    #     "iqr",
    #     _precomputed_weights_file=precomputed_weights,
    # )

    if dataset.metric == "gower":
        metric_func = gower
    elif dataset.metric == "bin":
        metric_func = bin_dist
    else:
        metric_func = dataset.metric

    print(
        f"----------------- Test using {dataset.metric} metric -----------------"
    )
    if dataset.task == "bin" or dataset.task == "multivar":
        enc = OrdinalEncoder()
        enc.set_params(encoded_missing_value=-1)

        cat_nom_cols = [
            i
            for i in range(len(gower.dtypes))
            if gower.dtypes[i] == DataType.CATEGORICAL_NOMINAL
        ] + [len(gower.dtypes)]
        fit_df = df[:, cat_nom_cols]

        enc.fit(fit_df)
        fit_df = enc.transform(fit_df)
        df[:, cat_nom_cols] = fit_df

        y = df[:, -1]
        df = df[:, :-1]

        df = np.ndarray.astype(df, dtype=np.float64)
        y = np.ndarray.astype(y, dtype=np.float64)

        train_set, test_set, y_train_set, y_test_set = train_test_split(
            df, y, test_size=0.2
        )

        if dataset.metric == "gower":
            gower.fit(train_set)

        knn = KNeighborsClassifier(n_neighbors=5, metric=metric_func)
        knn.fit(train_set, y_train_set)

        score = knn.score(test_set, y_test_set)

        print(f"KNN score: {score}")
        return score

    elif dataset.task == "reg":
        train_set, test_set = train_test_split(df, test_size=0.3)

        gower.fit(train_set)

        knn = NearestNeighbors(n_neighbors=5, metric=metric_func)
        knn.fit(train_set)
        print(knn.kneighbors(test_set, 5, False))

    elif dataset.task == "cluster":
        enc = OrdinalEncoder()
        enc.set_params(encoded_missing_value=-1)

        # Categorical Nominal columns
        cat_nom_cols = [
            i
            for i in range(len(gower.dtypes))
            if gower.dtypes[i] == DataType.CATEGORICAL_NOMINAL
        ]

        if dataset.labeled:
            cat_nom_cols += [len(gower.dtypes)]

        if len(cat_nom_cols) != 0:
            fit_df = df[:, cat_nom_cols]

            enc.fit(fit_df)
            fit_df = enc.transform(fit_df)
            df[:, cat_nom_cols] = fit_df

        y = None
        if dataset.labeled:
            y = df[:, -1]
            df = df[:, :-1]
            y = np.ndarray.astype(y, dtype=np.float64)

        df = np.ndarray.astype(df, dtype=np.float64)

        start = timeit.default_timer()
        if dataset.metric == "gower":
            gower.fit(df)
        print(f"Performing fit: {timeit.default_timer() - start}")

        # Hierarchical Clustering and dendrogram (without plotting)
        Z = linkage(df, method="average", metric=metric_func)
        # plt.figure()
        # dn = dendrogram(Z, no_plot=True)
        # plt.show()

        num_of_clusters = (
            gower.number_of_clusters_ if dataset.metric == "gower" else 3
        )

        start = timeit.default_timer()
        dist_x = pdist(df, metric=metric_func)
        print(f"Calculating dist matrix: {timeit.default_timer() - start}")
        pred_labels = fcluster(Z, t=num_of_clusters, criterion="maxclust")

        c, cophenetic_distances = cpcc(dist_x, Z)
        i = ioa(dist_x, cophenetic_distances)

        print(f"CPCC: {c}")
        print(f"IoA: {i}")

        if np.max(pred_labels) > 1:
            s = silhouette_score(df, pred_labels, metric=metric_func)
            cal_halab = calinski_harabasz_score(df, pred_labels)
            dav_bould = davies_bouldin_score(df, pred_labels)
            print(f"Silhouette: {s}")
            print(f"Calinski-Harabasz: {cal_halab}")
            print(f"Davies-Bouldin index: {dav_bould}")
            # silhouette_test(Z, df, metric_func)
            # pca_test(df, y, pred_labels)
        else:
            print("Predicted labels = 1!")

        # plt.title(dataset.metric)
        # plt.imshow(squareform(cophenetic_distances), cmap='hot')
        # plt.show()

    else:
        print("Wrong task!")
    print(
        "--------------------------------------------------------------------\n"
    )


def load_dataset(dataset_name: str):
    loaded_data = np.loadtxt(dataset_name, delimiter=",", dtype=object)
    cols_type = np.loadtxt(
        dataset_name[:-4] + "_cols_type.csv", delimiter=",", dtype=object
    )
    cols_type = np.array([Data.cols_type_maping[k] for k in cols_type])
    labels = loaded_data[1, :]
    loaded_data = loaded_data[1:, :]
    return loaded_data, cols_type, labels


def fill_na(data: np.array):
    data[data == ""] = -1
    return data


def fill_nan(data: np.array):
    data[np.isnan(data)] = -1
    return data


def load_sets():
    D_data = {}
    D_cols_type = {}
    D_labels = {}

    for file in listdir(os.path.abspath("datasets")):
        if (
            isfile(os.path.abspath("datasets") + "/" + file)
            and "_cols_type" not in file
        ):
            (
                D_data[file[:-4]],
                D_cols_type[file[:-4]],
                D_labels[file[:-4]],
            ) = load_dataset(os.path.abspath("datasets") + "/" + file)
    D = Data(D_data, D_cols_type, D_labels)
    return D


if __name__ == "__main__":

    D = load_sets()

    print(f"Loaded sets: {list(D.data.keys())}")

    test_dataset_name = "quakes"
    test_type = "cluster"
    labeled = False  # if dataset has column labels in same file as columns

    ds1 = Dataset(test_dataset_name, test_type, "gower", labeled)
    ds2 = Dataset(test_dataset_name, test_type, "bin", labeled)
    ds3 = Dataset(test_dataset_name, test_type, "euclidean", labeled)
    ds4 = Dataset(test_dataset_name, test_type, "cosine", labeled)
    ds5 = Dataset(test_dataset_name, test_type, "minkowski", labeled)
    ds6 = Dataset(test_dataset_name, test_type, "dice", labeled)
    ds7 = Dataset(test_dataset_name, test_type, "jaccard", labeled)

    n = 250

    gower = GowerMetric(
        D.cols_type[ds1.name],
        "iqr",
        weights="precomputed",
        precomputed_weights_file="gower_metric_saved_weights/saved_weights_quakes.csv",
    )

    gower2 = GowerMetric2(
        D.cols_type[ds1.name],
        "iqr",
        _precomputed_weights_file="gower_metric_saved_weights/saved_weights_quakes.csv",
    )

    print(
        "=========================== Vectorized ============================="
    )
    mertic_test(
        gower,
        ds1,
        D,
        n,
    )

    print(
        "========================= Not Vectorized ==========================="
    )
    mertic_test(
        gower2,
        ds1,
        D,
        n,
    )

    # mertic_test(ds2, D, n)
    # mertic_test(ds3, D, n)
    # mertic_test(ds4, D, n)
    # mertic_test(ds5, D, n)
    # mertic_test(ds6, D, n)
    # mertic_test(ds7, D, n)
