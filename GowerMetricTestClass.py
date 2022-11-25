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


class GowerMetric2:
    def __init__(
        self,
        dtypes: np.array,
        ratio_scale_normalization="range",
        _precomputed_weights_file="",
        _cpcc_threshold=None,
        _save_computed_weights=False,
    ):
        self.dtypes = dtypes  # initialize with np.array of column data types
        self.ranges_: List  # values of ranges in ._ratio_scale() (iqr or traditional range)
        self.h_: List  # h values in ._ratio_scale()
        self.n_features_in_: int
        self._sigma_sum = 0  # counter for existing values in vector
        self.ratio_scale_normalization: str = ratio_scale_normalization
        self._similarities_map = {
            DataType.BINARY_SYMMETRIC: self._bin_sym,
            DataType.CATEGORICAL_NOMINAL: self._cat_nom,
            DataType.RATIO_SCALE: self._ratio_scale,
        }

        self._cpcc_threshold = _cpcc_threshold
        self._precomputed_weights_file = _precomputed_weights_file
        self._save_computed_weights = _save_computed_weights

    def __call__(
        self,
        vector_1: np.array,
        vector_2: np.array,
    ):
        sklearn.utils.validation.check_is_fitted(self)

        if self.n_features_in_ != len(vector_1) | len(vector_2):
            print("Vector sizes don't match!")
            print(self.n_features_in_, len(vector_1), len(vector_2))
            return -1

        dist_func = lambda similarity, a, b: 1 - similarity(a, b)
        dist_func_ratio = lambda similarity, a, b, r, h: 1 - similarity(
            a, b, r, h
        )
        distances = [
            (
                dist_func(
                    self._similarities_map[self.dtypes[i]],
                    vector_1[i],
                    vector_2[i],
                )
                if self.dtypes[i] != DataType.RATIO_SCALE
                else dist_func_ratio(
                    self._similarities_map[self.dtypes[i]],
                    vector_1[i],
                    vector_2[i],
                    self.ranges_[i],
                    self.h_[i],
                )
            )
            * self.weights_[i]
            for i in range(self.n_features_in_)
        ]

        distance = np.sum(distances)

        distance /= self._sigma_sum
        self._reset_sigma()

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

        if not self._precomputed_weights_file:
            self._select_weights(X)
        else:
            self._load_weights(self._precomputed_weights_file)
        self._select_number_of_clusters(X)

    def _bin_sym(self, value_1, value_2) -> float:
        self._add_sigma()
        return 1.0 if np.isclose(value_1, value_2) else 0.0

    def _bin_asym(self, value_1, value_2) -> float:
        if np.isclose(value_1, value_2) and np.isclose(value_1, 1.0):
            self._add_sigma()
            return 1.0
        else:
            return 0.0

    def _cat_nom(self, value_1, value_2) -> float:
        self._add_sigma()
        return 1.0 if np.isclose(value_1, value_2) else 0.0

    def _ratio_scale(self, value_1, value_2, val_range, h) -> float:
        self._add_sigma()
        absolute = np.abs(value_1 - value_2)
        if absolute >= val_range:
            return 0.0
        elif absolute <= h:
            return 1.0
        else:
            return 1.0 - absolute / val_range

    def _add_sigma(self):
        self._sigma_sum += 1

    def _reset_sigma(self):
        self._sigma_sum = 0

    def _S_k(self, value_1, value_2, k):
        return (
            self._ratio_scale(
                value_1[k], value_2[k], self.ranges_[k], self.h_[k]
            )
            if self.dtypes[k] == DataType.RATIO_SCALE
            else self._cat_nom(value_1[k], value_2[k])
        )

    def _cpcc(self, weights, X, t=None, pbar=None):
        self.weights_ = weights
        x = pdist(X, metric=self)
        Z = linkage(x, method="average", metric=self)
        return -cophenet(Z, x)[0]

    def _cophenetic_dist(self, x):
        Z = linkage(x, method="average", metric=self)
        return cophenet(Z, x)

    def _cpcc_derivative(self, weights, X, t, pbar: tqdm):
        x = pdist(X, metric=self)
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
        pbar.update(5)
        # print('deriv:', np.array([derivative_cpcc_func(k) for k in range(self.n_features_in_)]))
        return np.array(
            [-derivative_cpcc_func(k) for k in range(self.n_features_in_)]
        )

    def _init_weights(self, k):
        return np.array(
            [
                [
                    np.random.uniform(
                        1 / (3 * self.n_features_in_),
                        1
                        - (self.n_features_in_ - 1)
                        / (3 * self.n_features_in_),
                    )
                    for _ in range(self.n_features_in_)
                ]
                for _ in range(k)
            ]
        )

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

    def _select_weights(self, X):
        with tqdm(total=100, ncols=100) as pbar:
            number_of_initial_weights = 10
            initial_weights = self._init_weights(number_of_initial_weights)

            X = X.copy()
            enc = OrdinalEncoder()
            enc.set_params(encoded_missing_value=-1)

            # Categorical Nominal columns
            cat_nom_cols = [
                i
                for i in range(len(self.dtypes))
                if self.dtypes[i] == DataType.CATEGORICAL_NOMINAL
            ]

            if len(cat_nom_cols) != 0:
                fit_X = X[:, cat_nom_cols]

                enc.fit(fit_X)
                fit_X = enc.transform(fit_X)
                X[:, cat_nom_cols] = fit_X

            X = np.ndarray.astype(X, dtype=np.float64)

            pbar.set_description("Initializing weights with CPCC scores")

            initial_weights_scores = []
            for i in range(number_of_initial_weights):
                self.weights_ = initial_weights[i]
                S = pdist(X, metric=self)
                initial_weights_scores.append(self._cophenetic_dist(S))
                pbar.update(100 // number_of_initial_weights)

            initial_weights_scores.sort(key=lambda x: x[1][0], reverse=True)

            pbar.reset()

            best_weights = (0.0,)

            for i in range(number_of_initial_weights):
                pbar.set_description(f"Maximizing CPCC (Iteration {i + 1})")

                self.weights_ = initial_weights[i]
                opt_weights = minimize(
                    self._cpcc,
                    self.weights_,
                    args=(X, initial_weights_scores[i][1], pbar),
                    method="L-BFGS-B",
                    jac=self._cpcc_derivative,
                    options={"gtol": 1e-04},
                    bounds=((0, 1) for _ in range(self.n_features_in_)),
                )

                if opt_weights.fun < best_weights[0]:
                    best_weights = (opt_weights.fun, opt_weights.x)

                if (
                    self._cpcc_threshold is not None
                    and best_weights[0] < -self._cpcc_threshold
                ):
                    break

                pbar.reset()

            self.weights_ = best_weights[1]
            self.weights_ /= np.sum(self.weights_)

            if self._save_computed_weights:
                self._save_weights()

    def _load_weights(self, file_name: str):
        self.weights_ = np.loadtxt(file_name, delimiter=",", dtype=np.float64)

    def _save_weights(self):
        rand_id = np.random.randint(0, 100000)
        converted_weights = np.asarray([list(self.weights_)])
        dest_file_name = "saved_weights_" + str(rand_id) + ".csv"
        np.savetxt(dest_file_name, converted_weights, delimiter=",")
        print(f"Weights saved in {dest_file_name}")
