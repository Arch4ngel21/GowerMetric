import numpy as np
from numba import njit, prange


@njit(fastmath=True)
def gower_metric_call_func_numba(
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
):
    assert n_features_in_ == len(vector_1)
    assert n_features_in_ == len(vector_2)

    if cat_nom_num > 0:
        cat_nom_cols_1 = vector_1[cat_nom_idx]
        cat_nom_cols_2 = vector_2[cat_nom_idx]

        cat_nom_dist = [0 for i in range(cat_nom_num)]
        for i in prange(cat_nom_num):
            if cat_nom_cols_1[i] == cat_nom_cols_2[i]:
                cat_nom_dist[i] = 0
            else:
                cat_nom_dist[i] = 1

        for i in prange(cat_nom_num):
            if np.isnan(cat_nom_cols_1[i]) & np.isnan(cat_nom_cols_2[i]):
                cat_nom_dist[i] = 1.0

        if weights is not None:
            cat_nom_dist_sum = 0
            for i in prange(cat_nom_num):
                cat_nom_dist_sum += cat_nom_dist[i] * weights[i]
        else:
            cat_nom_dist_sum = 0
            for i in prange(cat_nom_num):
                cat_nom_dist_sum += cat_nom_dist[i]
    else:
        cat_nom_dist_sum = 0.0

    if bin_asym_num > 0:
        bin_asym_cols_1 = vector_1[bin_asym_idx]
        bin_asym_cols_2 = vector_2[bin_asym_idx]

        # 0 if x1 == x2 == 1 or x1 != x2, so it's same as 1 if x1 == x2 == 0
        bin_asym_dist = [0 for i in range(bin_asym_num)]
        for i in prange(bin_asym_num):
            if bin_asym_cols_1[i] == 0 and bin_asym_cols_2[i] == 0:
                bin_asym_dist[i] = 1
            else:
                bin_asym_dist[i] = 0

        for i in prange(bin_asym_num):
            if np.isnan(bin_asym_cols_1[i]) & np.isnan(bin_asym_cols_2[i]):
                bin_asym_dist[i] = 1.0

        if weights is not None:
            bin_asym_dist_sum = 0
            for i in prange(bin_asym_num):
                bin_asym_dist_sum += bin_asym_dist[i] * weights[i]
        else:
            bin_asym_dist_sum = 0
            for i in prange(bin_asym_num):
                bin_asym_dist_sum += bin_asym_dist[i]
    else:
        bin_asym_dist_sum = 0.0

    if ratio_scale_num > 0:
        ratio_scale_cols_1 = vector_1[ratio_scale_idx]
        ratio_scale_cols_2 = vector_2[ratio_scale_idx]

        ratio_dist = [0 for i in range(ratio_scale_num)]
        for i in prange(ratio_scale_num):
            ratio_dist[i] = abs(ratio_scale_cols_1[i] - ratio_scale_cols_2[i])

        if ratio_scale_normalization == "iqr":
            above_threshold = [False for i in range(ratio_scale_num)]
            for i in prange(ratio_scale_num):
                above_threshold[i] = ratio_dist[i] >= ranges_[i]

        if ratio_scale_window == "kde":
            below_threshold = [False for i in range(ratio_scale_num)]
            for i in prange(ratio_scale_num):
                below_threshold[i] = ratio_dist[i] <= h_[i]

        ratio_dist_2 = [0 for i in range(ratio_scale_num)]
        for i in prange(ratio_scale_num):
            ratio_dist_2[i] = ratio_dist[i] / ranges_[i]

        for i in prange(ratio_scale_num):
            if np.isnan(ratio_scale_cols_1[i]) & np.isnan(ratio_scale_cols_2[i]):
                ratio_dist[i] = 1.0

        if ratio_scale_normalization == "iqr":
            for i in prange(ratio_scale_num):
                if above_threshold[i]:
                    ratio_dist_2[i] = 1.0

        if ratio_scale_window == "kde":
            for i in prange(ratio_scale_num):
                if below_threshold[i]:
                    ratio_dist_2[i] = 1.0

        if weights is not None:
            ratio_dist_sum = 0
            for i in prange(ratio_scale_num):
                ratio_dist_sum += ratio_dist_2[i] * weights[i]
        else:
            ratio_dist_sum = 0
            for i in prange(ratio_scale_num):
                ratio_dist_sum += ratio_dist_2[i]
    else:
        ratio_dist_sum = 0.0

    distance = cat_nom_dist_sum + bin_asym_dist_sum + ratio_dist_sum

    # Normalization
    distance /= n_features_in_

    return distance


