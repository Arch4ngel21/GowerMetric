import pytest
import numpy as np
from sklearn.preprocessing import OrdinalEncoder
from scipy.spatial.distance import pdist, squareform

from GowerMetric import GowerMetric
from utils import DataType

import pandas as pd


def test_cat_nom():
    data = np.array(
        [
            [
                "Private",
                "Assoc-acdm",
                "Never-married",
                "Sales",
                "Not-in-family",
                "Black",
                "Male",
                "United-States",
            ],
            [
                "Private",
                "Assoc-voc",
                "Married-civ-spouse",
                "Craft-repair",
                "Husband",
                "Asian-Pac-Islander",
                "Male",
                "?",
            ],
        ]
    )
    gower = GowerMetric(
        np.array(
            [DataType.CATEGORICAL_NOMINAL for _ in range(data.shape[1])]
        )
    )
    enc = OrdinalEncoder()
    enc.set_params(encoded_missing_value=-1)
    enc.fit(data)
    data = enc.transform(data)

    gower.fit(data)
    res = gower(data[0], data[1])
    assert res == 6 / 8


def test_ratio():
    data = np.array(
        [
            [35, 2, 2, 1, 0, 22, 24],
            [28, 2, 0, 1, 2, 23, 19],
            [36, 1, 0, 1, 1, 24, 12],
            [27, 2, 1, 4, 1, 25, 18],
            [40, 2, 0, 1, 2, 26, 27],
            [38, 2, 0, 1, 2, 27, 26],
            [34, 3, 0, 1, 2, 28, 31],
            [28, 4, 1, 3, 2, 29, 34],
        ]
    )
    # r: 13  3  2  3  2   7  22
    # d: 7/13 0 2/2 0 2/2 1/7 5/22
    gower = GowerMetric(
        np.array([DataType.RATIO_SCALE for _ in range(data.shape[1])])
    )
    gower.fit(data)
    res = gower(data[0], data[1])
    assert res == (7 / 13 + 1 + 1 + 1 / 7 + 5 / 22) / 7


def test_bin_asym():
    data = np.array([[1, 1, 0, 1, 1], [1, 0, 0, 0, 1]])
    gower = GowerMetric(
        np.array(
            [DataType.BINARY_ASYMMETRIC for _ in range(data.shape[1])]
        )
    )
    gower.fit(data)
    res = gower(data[0], data[1])
    assert res == 1 / 5


def test_weights():
    data = np.array(
        [
            [17.97, 181.66, 626, 4.1, 19],
            [20.42, 181.96, 649, 4, 11],
            [19.68, 184.31, 195, 4, 12],
            [11.7, 166.1, 82, 4.8, 43],
            [28.11, 181.93, 194, 4.4, 15],
            [28.74, 181.74, 211, 4.7, 35],
            [17.47, 179.59, 622, 4.3, 19],
        ]
    )
    # r: 17.04 18.21 567 0.8 32
    # d: 2.45 0.3 23 0.1 8
    gower_weights = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    dist = np.array([2.45, 0.3, 23.0, 0.1, 8.0])
    ranges = np.array([17.04, 18.21, 567.0, 0.8, 32.0])
    gower = GowerMetric(
        dtypes=np.array([DataType.RATIO_SCALE for _ in range(data.shape[1])]),
        weights=gower_weights,
    )
    gower.fit(data)
    res = gower(data[0], data[1])
    assert res == ((dist / ranges) @ gower_weights) / 5.0


def test_all_types():
    data = np.array(
        [
            ['Poland', 21, 5, 10000, 0, 1, 1],
            ['Germany', 50, 4, 20000, 0, 1, 0],
            ['Poland', 32, 3, 15000, 0, 0, 0],
            ['France', 40, 4, 25000, 0, 1, 1],
            ['Denmark', 45, 4, 23000, 0, 1, 1]
        ]
    )
    # ranges: 29, 2, 15000
    # dist: 1, 29, 1, 10000, 1, 0, 0
    ranges = np.array([1, 29, 2, 15000, 1, 1, 1])
    dist = np.array([1, 29, 1, 10000, 0, 0, 0])

    gower = GowerMetric(
        dtypes=np.array(
            [DataType.CATEGORICAL_NOMINAL,
             DataType.RATIO_SCALE,
             DataType.RATIO_SCALE,
             DataType.RATIO_SCALE,
             DataType.BINARY_SYMMETRIC,
             DataType.BINARY_ASYMMETRIC,
             DataType.BINARY_ASYMMETRIC]
        )
    )

    enc = OrdinalEncoder()
    enc.set_params(encoded_missing_value=-1)
    enc.fit(data[:, [0]])
    data[:, [0]] = enc.transform(data[:, [0]])
    data = np.ndarray.astype(data, dtype=np.float64)

    gower.fit(data)
    res = gower(data[0], data[1])

    assert np.isclose(res, np.sum(dist / ranges) / 7.0)


def test_negative_values():
    data = np.array(
        [
            [-17.97, -181.66, -626, -4.1, -19],
            [-20.42, -181.96, -649, -4, -11],
            [-19.68, -184.31, -195, -4, -12],
            [-11.7, -166.1, -82, -4.8, -43],
            [-28.11, -181.93, -194, -4.4, -15],
            [-28.74, -181.74, -211, -4.7, -35],
            [-17.47, -179.59, -622, -4.3, -19]
         ]
    )

    gower = GowerMetric(
        dtypes=np.array([DataType.RATIO_SCALE for _ in range(data.shape[1])]),
    )
    dist = np.array([2.45, 0.3, 23.0, 0.1, 8.0])
    ranges = np.array([17.04, 18.21, 567.0, 0.8, 32.0])

    gower.fit(data)
    res = gower(data[0], data[1])
    assert np.isclose(res, np.sum(dist / ranges) / 5.0)


def test_check_non_zero_on_diagonal():
    data = np.array(
        [
            ["25 - 34", "40 - 79", "19 - 10", 0, 7],
            ["25 - 34", "40 - 79", "20 - 29", 0, 4],
            ["25 - 34", "40 - 79", "30 +", 0, 7],
            ["25 - 34", "80 - 119", "0 - 9g / day", 0, 2],
            ["25 - 34", "80 - 119", "19 - 10", 0, 1],
            ["25 - 34", "80 - 119", "30 +", 0, 2],
            ["25 - 34", "120 +", "0 - 9g / day", 0, 1],
            ["25 - 34", "120 +", "19 - 10", 1, 0],
            ["25 - 34", "120 +", "20 - 29", 0, 1],
            ["25 - 34", "120 +", "30 +", 0, 2],
            ["35 - 44", "0 - 39g / day", "0 - 9g / day", 0, 60],
            ["35 - 44", "0 - 39g / day", "19 - 10", 1, 13],
            ["35 - 44", "0 - 39g / day", "20 - 29", 0, 7],
            ["35 - 44", "0 - 39g / day", "30 +", 0, 8],
            ["35 - 44", "40 - 79", "0 - 9g / day", 0, 35],
            ["35 - 44", "40 - 79", "19 - 10", 3, 20],
            ["35 - 44", "40 - 79", "20 - 29", 1, 13]
        ]
    )

    gower = GowerMetric(
        dtypes=np.array(
            [DataType.CATEGORICAL_NOMINAL,
             DataType.CATEGORICAL_NOMINAL,
             DataType.CATEGORICAL_NOMINAL,
             DataType.RATIO_SCALE,
             DataType.RATIO_SCALE]
        )
    )

    enc = OrdinalEncoder()
    enc.set_params(encoded_missing_value=-1)
    enc.fit(data)
    data = enc.transform(data)

    gower.fit(data)
    dist_matrix = squareform(pdist(data, metric=gower))

    assert np.sum(np.diagonal(dist_matrix)) == 0


def test_nan_max_dist():
    data = pd.DataFrame(
        [
            [np.nan, 21, 5, 10000, 0, 1, 1],
            [np.nan, 50, 4, 20000, 0, np.nan, np.nan],
            ['Poland', 32, 3, np.nan, 0, 0, 0],
            ['France', np.nan, 4, 25000, 0, 1, 1],
            ['Denmark', 45, 4, 23000, 0, 1, 1]
        ]
    )

    # ranges: 29, 2, 15000
    # dist: 1, 29, 1, 10000, 1, 0, 0
    ranges = np.array([1, 29, 2, 15000, 1, 1, 1])
    dist = np.array([1, 29, 1, 10000, 1, 0, 0])

    gower = GowerMetric(
        dtypes=np.array(
            [DataType.CATEGORICAL_NOMINAL,
             DataType.RATIO_SCALE,
             DataType.RATIO_SCALE,
             DataType.RATIO_SCALE,
             DataType.BINARY_SYMMETRIC,
             DataType.BINARY_SYMMETRIC,
             DataType.BINARY_ASYMMETRIC]
        ),
        nan_values_handling="max_dist"
    )

    enc = OrdinalEncoder(
        categories="auto",
        dtype=np.float64,
        handle_unknown="use_encoded_value",
        unknown_value=np.nan,
        encoded_missing_value=np.nan
    )
    enc.fit(data.iloc[:, [0]])
    data.iloc[:, [0]] = enc.transform(data.iloc[:, [0]])
    data = data.to_numpy(dtype=np.float64)
    # print('\n', data)

    gower.fit(data)
    res = gower(data[0], data[1])

    assert np.isclose(res, np.sum(dist / ranges) / 7.0)


def test_nan_raise():
    data = pd.DataFrame(
        [
            [np.nan, 21, 5, 10000, 0, 1, 1],
            [np.nan, 50, 4, 20000, 0, np.nan, np.nan],
            ['Poland', 32, 3, np.nan, 0, 0, 0],
            ['France', np.nan, 4, 25000, 0, 1, 1],
            ['Denmark', 45, 4, 23000, 0, 1, 1]
        ]
    )

    # ranges: 29, 2, 15000
    # dist: 1, 29, 1, 10000, 1, 0, 0
    ranges = np.array([1, 29, 2, 15000, 1, 1, 1])
    dist = np.array([1, 29, 1, 10000, 0, 1, 1])

    gower = GowerMetric(
        dtypes=np.array(
            [DataType.CATEGORICAL_NOMINAL,
             DataType.RATIO_SCALE,
             DataType.RATIO_SCALE,
             DataType.RATIO_SCALE,
             DataType.BINARY_SYMMETRIC,
             DataType.BINARY_SYMMETRIC,
             DataType.BINARY_ASYMMETRIC]
        )
    )

    enc = OrdinalEncoder(
        categories="auto",
        dtype=np.float64,
        handle_unknown="use_encoded_value",
        unknown_value=np.nan,
        encoded_missing_value=np.nan
    )
    enc.fit(data.iloc[:, [0]])
    data.iloc[:, [0]] = enc.transform(data.iloc[:, [0]])
    data = data.to_numpy(dtype=np.float64)

    try:
        gower.fit(data)
        res = gower(data[0], data[1])
    except ValueError:
        assert True
        exit(0)
    assert False


def test_nan_ignore():
    data = pd.DataFrame(
        [
            [np.nan, 21, 5, 10000, 0, 1, 1],
            [np.nan, 50, 4, 20000, 0, np.nan, np.nan],
            ['Poland', 32, 3, np.nan, 0, 0, 0],
            ['France', np.nan, 4, 25000, 0, 1, 1],
            ['Denmark', 45, 4, 23000, 0, 1, 1]
        ]
    )

    # ranges: 29, 2, 15000
    # dist: 0, 29, 1, 10000, 1, 0, 0
    ranges = np.array([1, 29, 2, 15000, 1, 1, 1])
    dist = np.array([0, 29, 1, 10000, 0, 0, 0])

    gower = GowerMetric(
        dtypes=np.array(
            [DataType.CATEGORICAL_NOMINAL,
             DataType.RATIO_SCALE,
             DataType.RATIO_SCALE,
             DataType.RATIO_SCALE,
             DataType.BINARY_SYMMETRIC,
             DataType.BINARY_SYMMETRIC,
             DataType.BINARY_ASYMMETRIC]
        ),
        nan_values_handling="ignore"
    )

    enc = OrdinalEncoder(
        categories="auto",
        dtype=np.float64,
        handle_unknown="use_encoded_value",
        unknown_value=np.nan,
        encoded_missing_value=np.nan
    )
    enc.fit(data.iloc[:, [0]])
    data.iloc[:, [0]] = enc.transform(data.iloc[:, [0]])
    data = data.to_numpy(dtype=np.float64)
    # print('\n', data)

    gower.fit(data)
    res = gower(data[0], data[1])

    assert np.isclose(res, np.sum(dist / ranges) / 4.0)
