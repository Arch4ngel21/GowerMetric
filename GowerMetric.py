import os.path
import timeit
from pathlib import Path
from typing import List, Optional

import numpy as np
import sklearn.utils.validation
from scipy.spatial.distance import pdist
from scipy.cluster.hierarchy import (
    linkage,
    cophenet,
    fcluster,
)
from scipy.optimize import minimize
from sklearn.metrics import (
    silhouette_score,
)
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from tqdm import tqdm

from utils import DataType


class GowerMetric:
    def __init__(
        self,
        dtypes: np.array,
        ratio_scale_normalization="range",
        weights=None,
        precomputed_weights_file=None,
    ):
        self.dtypes = dtypes  # initialize with np.array of column data types
        self.weights = weights
        self.precomputed_weights_file = precomputed_weights_file
        self.ranges_: List  # values of ranges in .ratio_scale() (iqr or traditional range)
        self.h_: List  # h values in .ratio_scale()
        self.n_features_in_: int
        self.ratio_scale_normalization: str = ratio_scale_normalization

        self._cat_nom_idx = [
            True
            if self.dtypes[i] == DataType.CATEGORICAL_NOMINAL
            or self.dtypes[i] == DataType.BINARY_SYMMETRIC
            else False
            for i in range(len(self.dtypes))
        ]
        self._ratio_scale_idx = [
            True if self.dtypes[i] == DataType.RATIO_SCALE else False
            for i in range(len(self.dtypes))
        ]

        # Sums of columns of given type
        self._cat_nom_num = sum(self._cat_nom_idx)
        self._ratio_scale_num = sum(self._ratio_scale_idx)

    def __call__(
        self,
        vector_1: np.ndarray,
        vector_2: np.ndarray,
    ):
        # TODO - no check for fit
        # sklearn.utils.validation.check_is_fitted(self)
        if self.n_features_in_ != len(vector_1) | len(vector_2):
            assert len(vector_1) == len(vector_2)

        # Distance for cat_nom and bin_sym columns
        if self._cat_nom_num:
            cat_nom_cols_1 = vector_1[self._cat_nom_idx]
            cat_nom_cols_2 = vector_2[self._cat_nom_idx]
            cat_nom_dist = 1.0 - (cat_nom_cols_1 == cat_nom_cols_2)

            # Weighted variant - dot product with according self.weights columns
            # Unweighted - a sum of distances of between all columns
            if self.weights is not None:
                cat_nom_dist = cat_nom_dist @ self.weights[self._cat_nom_idx]
            else:
                cat_nom_dist = cat_nom_dist.sum()
        else:
            cat_nom_dist = 0.0

        # Distance for ratio_scale columns
        if self._ratio_scale_num:
            ratio_scale_cols_1 = vector_1[self._ratio_scale_idx]
            ratio_scale_cols_2 = vector_2[self._ratio_scale_idx]
            ratio_dist = np.abs(ratio_scale_cols_1 - ratio_scale_cols_2)

            ones = ratio_dist >= self.ranges_[self._ratio_scale_idx]
            zeros = ratio_dist <= self.h_[self._ratio_scale_idx]

            if ratio_scale_cols_1.size == 1:
                if ones:
                    ratio_dist = 1.0
                elif zeros:
                    ratio_dist = 0.0
                else:
                    ratio_dist /= self.ranges_[self._ratio_scale_idx]
            else:
                ratio_dist /= self.ranges_[self._ratio_scale_idx]
                ratio_dist[zeros] = 0.0
                ratio_dist[ones] = 1.0

            if self.weights is not None:
                ratio_dist = ratio_dist @ self.weights[self._ratio_scale_idx]
            else:
                ratio_dist = ratio_dist.sum()
        else:
            ratio_dist = 0.0

        # Sum of all distances
        distance = cat_nom_dist + ratio_dist

        # Normalization
        distance /= self.n_features_in_

        return distance

    def fit(self, X):
        self.ranges_ = np.zeros(self.dtypes.size)
        self.h_ = np.zeros(self.dtypes.size)
        self.n_features_in_ = self.dtypes.size

        for i in range(self.dtypes.size):
            if self.dtypes[i] == DataType.RATIO_SCALE:
                column = X[:, i]

                if self.ratio_scale_normalization == "iqr":
                    # IQR (g_t) - Interquartile Range
                    q1, q3 = np.percentile(column, [25, 75])
                    self.ranges_[i] = q3 - q1

                    if self.ranges_[i] == 0:
                        self.ranges_[i] = np.ptp(column, axis=0)

                elif self.ratio_scale_normalization == "range":
                    # Traditional range of values
                    self.ranges_[i] = np.ptp(column, axis=0)
                else:
                    print(
                        "GowerMetric - Ratio scale has wrong type of normalization!"
                    )
                    raise ValueError

                # h_t - bandwidth in the kernel density estimation (Marcello D’Orazio - p. 9)
                c = 1.06

                s = np.std(column)
                n = X[:, i].size
                self.h_[i] = (
                    c / n ** (1 / 5) * np.min([s, self.ranges_[i] / 1.34])
                )

        if self.weights == "precomputed":
            loader = GowerMetricWeightsLoader(self)
            loader.load_weights(self.precomputed_weights_file)
        elif self.weights == "cpcc":
            loader = GowerMetricWeightsLoader(self)
            loader.select_weights(X)

        self._select_number_of_clusters(X)

    def cat_nom_bin_sym(
        self, vector_1: np.ndarray, vector_2: np.ndarray
    ) -> np.ndarray:

        return vector_1 == vector_2

    def ratio_scale(
        self,
        vector_1: np.ndarray,
        vector_2: np.ndarray,
        val_range: np.ndarray,
        h: np.ndarray,
    ) -> np.ndarray:

        absolute = vector_1 - vector_2
        absolute = np.abs(absolute)

        ones = absolute >= val_range
        zeros = absolute <= h

        if vector_1.size == 1:
            if ones:
                return np.array(1.0)
            elif zeros:
                return np.array(0.0)
            else:
                absolute /= val_range
        else:
            absolute /= val_range
            absolute[zeros] = 0.0
            absolute[ones] = 1.0

        return 1.0 - absolute

    def _select_number_of_clusters(self, X):
        Z = linkage(X, method="average", metric=self)

        # So scores[i] refers to score for max i clusters
        scores = [0.0, 0.0, 0.0]

        for i in range(3, 10):
            pred_labels = fcluster(Z, t=i, criterion="maxclust")
            if len(np.unique(pred_labels)) > 1:
                scores.append(silhouette_score(X, pred_labels, metric=self))
            else:
                scores.append(0.0)
        scores = np.array(scores)

        self.number_of_clusters_ = np.argmax(scores)

        # X_values = np.linspace(3, 10, 7)
        # plt.plot(X_values, scores[3:])
        # plt.show()


class GowerMetricWeightsLoader:
    def __init__(
        self,
        gower: GowerMetric,
        _precomputed_weights_file="",
        _cpcc_threshold=None,
        _save_computed_weights=True,
    ):
        self.gower = gower
        self._precomputed_weights_file = _precomputed_weights_file
        self._cpcc_threshold = _cpcc_threshold
        self._save_computed_weights = _save_computed_weights

    def _S_k(self, vector_1: np.ndarray, vector_2: np.ndarray, k: np.int64):
        return (
            self.gower.ratio_scale(
                np.array(vector_1[k]),
                np.array(vector_2[k]),
                np.array(self.gower.ranges_[k]),
                np.array(self.gower.h_[k]),
            )
            if self.gower.dtypes[k] == DataType.RATIO_SCALE
            else self.gower.cat_nom_bin_sym(vector_1[k], vector_2[k])
        )

    def _cpcc(self, weights, X, t=None, pbar=None):
        self.gower.weights_ = weights
        x = pdist(X, metric=self.gower)
        Z = linkage(x, method="average", metric=self.gower)
        return -cophenet(Z, x)[0]

    def _cophenetic_dist(self, x):
        Z = linkage(x, method="average", metric=self.gower)
        return cophenet(Z, x)

    def _cpcc_jac(self, weights, X, t):
        x = pdist(X, metric=self.gower)
        x_ = np.average(x)
        t_ = np.average(t)

        a = x - x_
        b = t - t_

        # Distance matrix calculated only for k feature
        S_i_j_k = lambda k: pdist(X, metric=self._S_k, k=k)

        # Derivative of CPCC as a function of k
        derivative_cpcc_func = lambda k: -np.sum(
            S_i_j_k(k)
            * (
                (b - a * np.sum(a * b) / np.sum(np.power(a, 2)))
                / np.sqrt(np.sum(np.power(a, 2)) * np.sum(np.power(b, 2)))
            )
        )
        return np.array(
            [
                -derivative_cpcc_func(k)
                for k in range(self.gower.n_features_in_)
            ]
        )

    def _init_weights(self, k):
        return np.array(
            [
                [
                    np.random.uniform(
                        1 / (3 * self.gower.n_features_in_),
                        1
                        - (self.gower.n_features_in_ - 1)
                        / (3 * self.gower.n_features_in_),
                    )
                    for _ in range(self.gower.n_features_in_)
                ]
                for _ in range(k)
            ]
        )

    def select_weights(self, X):
        number_of_initial_weights = 10
        initial_weights = self._init_weights(number_of_initial_weights)

        X = X.copy()
        enc = OrdinalEncoder()
        enc.set_params(encoded_missing_value=-1)

        # Categorical Nominal columns
        cat_nom_cols = [
            i
            for i in range(len(self.gower.dtypes))
            if self.gower.dtypes[i] == DataType.CATEGORICAL_NOMINAL
        ]

        if len(cat_nom_cols) != 0:
            fit_X = X[:, cat_nom_cols]

            enc.fit(fit_X)
            fit_X = enc.transform(fit_X)
            X[:, cat_nom_cols] = fit_X

            X = np.ndarray.astype(X, dtype=np.float64)

        initial_weights_scores = []
        for i in range(number_of_initial_weights):
            self.gower.weights_ = initial_weights[i]
            S = pdist(X, metric=self.gower)
            initial_weights_scores.append(self._cophenetic_dist(S))

        initial_weights_scores.sort(key=lambda x: x[1][0], reverse=True)

        best_weights = (0.0,)

        for i in range(number_of_initial_weights):
            self.gower.weights_ = initial_weights[i]
            opt_weights = minimize(
                self._cpcc,
                self.gower.weights_,
                args=(X, initial_weights_scores[i][1]),
                method="L-BFGS-B",
                jac=self._cpcc_jac,
                options={"gtol": 1e-04},
                bounds=((0, 1) for _ in range(self.gower.n_features_in_)),
            )

            if opt_weights.fun < best_weights[0]:
                best_weights = (opt_weights.fun, opt_weights.x)

            if (
                self._cpcc_threshold is not None
                and best_weights[0] < -self._cpcc_threshold
            ):
                break

        self.gower.weights_ = best_weights[1]
        self.gower.weights_ /= np.sum(self.gower.weights_)

        if self._save_computed_weights:
            self._save_weights()

    def load_weights(self, file_name: str):
        self.gower.weights = np.loadtxt(
            file_name, delimiter=",", dtype=np.float64
        )

    def _save_weights(self):
        weights_dir_name = (
            str(Path().absolute()) + "\\gower_metric_saved_weights"
        )
        if not os.path.exists(weights_dir_name):
            os.mkdir(weights_dir_name)

        rand_id = np.random.randint(0, 100000)
        # converted_weights = np.asarray([list(self.gower.weights)])
        dest_file_name = (
            weights_dir_name + "\\saved_weights_" + str(rand_id) + ".csv"
        )
        np.savetxt(dest_file_name, self.gower.weights, delimiter=",")
        print(f"Weights saved in {dest_file_name}")
