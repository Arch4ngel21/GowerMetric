from typing import Optional, Union

import numpy as np
from numba import njit

from utils import DataType
from weights import GowerMetricWeights


@njit
def gower_metric_call_func(
    vector_1: np.ndarray,
    vector_2: np.ndarray,
    weights: np.ndarray,
    cat_nom_num: int,
    bin_asym_num: int,
    ratio_scale_num: int,
    cat_nom_idx: np.ndarray,
    bin_asym_idx: np.ndarray,
    ratio_scale_idx: np.ndarray,
    ratio_scale_normalization: str,
    ratio_scale_window: str,
    ranges_: np.ndarray,
    h_: np.ndarray,
    n_features_in_: int,
    nan_values_handling: str
):
    assert n_features_in_ == len(vector_1)
    assert n_features_in_ == len(vector_2)

    cat_nom_ignored_num, bin_asym_ignored_num, ratio_scale_ignored_num = 0, 0, 0

    if cat_nom_num > 0:
        cat_nom_cols_1 = vector_1[cat_nom_idx]
        cat_nom_cols_2 = vector_2[cat_nom_idx]

        cat_nom_dist = 1.0 - (cat_nom_cols_1 == cat_nom_cols_2)

        # Handling nan values
        if nan_values_handling == "raise":
            if (np.isnan(cat_nom_cols_1) | np.isnan(cat_nom_cols_2)).any():
                raise ValueError
        elif nan_values_handling == "ignore":
            cat_nom_ignored = np.isnan(cat_nom_cols_1) | np.isnan(cat_nom_cols_2)
            cat_nom_ignored_num = np.sum(cat_nom_ignored)
            cat_nom_dist[cat_nom_ignored] = 0.0
        elif nan_values_handling == "mas_dist":
            cat_nom_dist[np.isnan(cat_nom_cols_1) | np.isnan(cat_nom_cols_2)] = 1.0

        if weights is not None:
            cat_nom_dist = cat_nom_dist @ weights[cat_nom_idx]
        else:
            cat_nom_dist = cat_nom_dist.sum()
    else:
        cat_nom_dist = 0.0

    if bin_asym_num > 0:
        bin_asym_cols_1 = vector_1[bin_asym_idx]
        bin_asym_cols_2 = vector_2[bin_asym_idx]

        # 0 if x1 == x2 == 1 or x1 != x2, so it's same as 1 if x1 == x2 == 0
        bin_asym_dist = np.asarray(
            (bin_asym_cols_1 == 0) & (bin_asym_cols_2 == 0), dtype=np.float64
        )

        # Handling nan values
        if nan_values_handling == "raise":
            if (np.isnan(bin_asym_cols_1) | np.isnan(bin_asym_cols_2)).any():
                raise ValueError
        elif nan_values_handling == "ignore":
            bin_asym_ignored = np.isnan(bin_asym_cols_1) | np.isnan(bin_asym_cols_2)
            bin_asym_ignored_num = np.sum(bin_asym_ignored)
            bin_asym_dist[bin_asym_ignored] = 0.0
        elif nan_values_handling == "mas_dist":
            bin_asym_dist[
                np.isnan(bin_asym_cols_1) | np.isnan(bin_asym_cols_2)
            ] = 1.0

        if weights is not None:
            bin_asym_dist = bin_asym_dist @ weights[bin_asym_idx]
        else:
            bin_asym_dist = bin_asym_dist.sum()
    else:
        bin_asym_dist = 0.0

    if ratio_scale_num > 0:
        ratio_scale_cols_1 = vector_1[ratio_scale_idx]
        ratio_scale_cols_2 = vector_2[ratio_scale_idx]
        ratio_dist = np.abs(ratio_scale_cols_1 - ratio_scale_cols_2)

        if ratio_scale_normalization == "iqr":
            above_threshold = ratio_dist >= ranges_

        if ratio_scale_window == "kde":
            below_threshold = ratio_dist <= h_

        ratio_dist = ratio_dist / ranges_

        # Handling nan values
        if nan_values_handling == "raise":
            if (np.isnan(ratio_scale_cols_1) | np.isnan(ratio_scale_cols_2)).any():
                raise ValueError
        elif nan_values_handling == "ignore":
            ratio_scale_ignored = np.isnan(ratio_scale_cols_1) | np.isnan(ratio_scale_cols_2)
            ratio_scale_ignored_num = np.sum(ratio_scale_ignored)
            ratio_dist[ratio_scale_ignored] = 0.0
        elif nan_values_handling == "mas_dist":
            ratio_dist[
                np.isnan(ratio_scale_cols_1) | np.isnan(ratio_scale_cols_2)
                ] = 1.0

        if ratio_scale_normalization == "iqr":
            ratio_dist[above_threshold] = 1.0

        if ratio_scale_window == "kde":
            ratio_dist[below_threshold] = 0.0

        if weights is not None:
            ratio_dist = ratio_dist @ weights[ratio_scale_idx]
        else:
            ratio_dist = ratio_dist.sum()
    else:
        ratio_dist = 0.0

    distance = cat_nom_dist + bin_asym_dist + ratio_dist

    # Normalization
    distance /= (n_features_in_ - cat_nom_ignored_num - bin_asym_ignored_num - ratio_scale_ignored_num)

    return distance


class GowerMetric:
    def __init__(
        self,
        dtypes: np.array,
        ratio_scale_normalization: str = "range",
        ratio_scale_window: Optional[str] = None,
        kde_type: Optional[str] = None,
        weights: Optional[Union[list, str, np.ndarray]] = None,
        precomputed_weights_file: Optional[str] = None,
        nan_values_handling: str = "raise",
        verbose: int = 0,
    ):
        assert (
            weights is None
            or weights == "precomputed"
            or weights == "cpcc"
            or type(weights) == np.ndarray
            or type(weights) == list
        )
        assert (
            ratio_scale_normalization == "range"
            or ratio_scale_normalization == "iqr"
        )
        assert ratio_scale_window is None or ratio_scale_window == "kde"
        assert (
            kde_type is None
            or kde_type == "cv_grid"
            or kde_type == "cv_optuna"
            or kde_type == "silverman"
            or kde_type == "scott"
            or kde_type == "sheather-jones"
        )
        assert (
            nan_values_handling == "raise",
            nan_values_handling == "ignore",
            nan_values_handling == "max_dist"
        )

        self.dtypes = dtypes  # initialize with np.array of column data types
        self.weights = weights
        self.precomputed_weights_file = precomputed_weights_file
        self.verbose = verbose
        self.nan_values_handling = nan_values_handling
        self.ranges_: np.ndarray  # values of ranges in .ratio_scale() (iqr or traditional range)
        self.h_: np.ndarray  # h values in .ratio_scale()
        self.n_features_in_: int
        self.ratio_scale_normalization: str = ratio_scale_normalization
        self.ratio_scale_window: str = ratio_scale_window
        self.kde_type: str = kde_type

        # Bit masks for certain column types
        self.ratio_scale_idx = self.dtypes == DataType.RATIO_SCALE
        self.cat_nom_idx = (self.dtypes == DataType.CATEGORICAL_NOMINAL) | (
            self.dtypes == DataType.BINARY_SYMMETRIC
        )
        self.bin_asym_idx = self.dtypes == DataType.BINARY_ASYMMETRIC

        # Sums of columns of given type
        self.cat_nom_num = np.sum(self.cat_nom_idx)
        self.ratio_scale_num = np.sum(self.ratio_scale_idx)
        self.bin_asym_num = np.sum(self.bin_asym_idx)

    def fit(self, X):
        assert X.shape[1] == len(self.dtypes)

        self.n_features_in_ = X.shape[1]
        self.ranges_ = np.ndarray([])
        self.h_ = np.ndarray([])

        if self.ratio_scale_num > 0:
            ratio_cols = X[:, self.ratio_scale_idx]

            col_mean = np.nanmean(ratio_cols, axis=0)
            nan_indices = np.where(np.isnan(ratio_cols))

            # print("nan_indices:", nan_indices, '\n')
            # print(ratio_cols, '\n')
            # print(ratio_cols[nan_indices], '\n')

            if self.nan_values_handling == "raise":
                if len(nan_indices) > 0:
                    raise ValueError

            ratio_cols[nan_indices] = np.take(col_mean, nan_indices[1])

            # g_t parameter
            if self.ratio_scale_normalization == "range":
                self.ranges_ = np.ptp(ratio_cols, axis=0)

            elif self.ratio_scale_normalization == "iqr":
                from scipy.stats import iqr

                self.ranges_ = iqr(ratio_cols, axis=0)

                # Needs this check
                zero_values_mask = self.ranges_ == 0
                self.ranges_[zero_values_mask] = np.ptp(
                    ratio_cols[:, zero_values_mask], axis=0
                )

            n = X.shape[0]

            # h_t parameter
            if self.ratio_scale_window == "kde":
                if self.kde_type == "silverman":
                    c = 1.06

                    s = np.std(ratio_cols, axis=0)
                    self.h_ = (
                        c
                        / n ** (1 / 5)
                        * np.min([s, self.ranges_ / 1.34], axis=0)
                    )
                elif self.kde_type == "scott":
                    c = 0.9

                    s = np.std(ratio_cols, axis=0)
                    self.h_ = c / n ** (1 / 5) * s
                elif self.kde_type == "sheather-jones":
                    from KDEpy import FFTKDE

                    try:
                        self.h_ = np.array(
                            [
                                FFTKDE(kernel="gaussian", bw="ISJ")
                                .fit(ratio_cols[:, i])
                                .bw
                                for i in range(self.ratio_scale_num)
                            ]
                        )
                    except ValueError:
                        print(
                            "FFTKDE resulted in failure - "
                            "Crashed with error: ValueError: Root finding did not converge. Need more data. - "
                            "Repeating fit using kde_type == 'silverman'"
                        )

                        # Try to fit using silverman instead
                        self.kde_type = "silverman"
                        self.fit(X)
                        return
                elif self.kde_type == "cv_grid":
                    from sklearn.model_selection import GridSearchCV
                    from sklearn.neighbors import KernelDensity

                    self.h_ = np.array(
                        [
                            GridSearchCV(
                                KernelDensity(),
                                {"bandwidth": np.linspace(0.1, 1e06, 100)},
                                cv=20,
                            )
                            .fit(ratio_cols[:, i].reshape(-1, 1))
                            .best_estimator_.bandwidth
                            for i in range(self.ratio_scale_num)
                        ]
                    )
                elif self.kde_type == "cv_optuna":
                    import optuna
                    from optuna.integration import OptunaSearchCV
                    from sklearn.neighbors import KernelDensity

                    self.h_ = []
                    for i in range(self.ratio_scale_num):
                        clf = KernelDensity()
                        param_distributions = {
                            "bandwidth": optuna.distributions.FloatDistribution(
                                1e-08, 1e06, log=True
                            )
                        }
                        optuna_search = OptunaSearchCV(
                            clf,
                            param_distributions,
                            cv=20,
                            n_trials=100,
                            verbose=0,
                        )
                        optuna_search.fit(ratio_cols[:, i].reshape(-1, 1))
                        self.h_.append(optuna_search.best_estimator_.bandwidth)
                    self.h_ = np.array(self.h_, dtype=np.float64)

        loader = GowerMetricWeights(self, _save_computed_weights=False)
        if self.weights == "precomputed":
            loader.load_weights(self.precomputed_weights_file)
        elif self.weights == "cpcc":
            self.weights = np.ones(self.n_features_in_)
            loader.select_weights(X)

        if self.weights is not None:
            loader.select_number_of_clusters(X)

    def __call__(
        self,
        vector_1: np.ndarray,
        vector_2: np.ndarray,
    ) -> np.float64:

        return gower_metric_call_func(
            vector_1,
            vector_2,
            self.weights,
            self.cat_nom_num,
            self.bin_asym_num,
            self.ratio_scale_num,
            self.cat_nom_idx,
            self.bin_asym_idx,
            self.ratio_scale_idx,
            self.ratio_scale_normalization,
            self.ratio_scale_window,
            self.ranges_,
            self.h_,
            self.n_features_in_,
            self.nan_values_handling
        )
