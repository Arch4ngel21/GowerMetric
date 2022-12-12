import unittest

import numpy as np

from GowerMetric import GowerMetric
from utils import DataType


class GowerMetricTest(unittest.TestCase):
    def test_cat_nom(self):
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
        gower.fit(data)
        res = gower(data[0], data[1])
        self.assertEqual(res, 6 / 8)

    def test_ratio(self):
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
        self.assertEqual(res, (7 / 13 + 1 + 1 + 1 / 7 + 5 / 22) / 7)

    def test_bin_asym(self):
        data = np.array([[1, 1, 0, 1, 1], [1, 0, 0, 0, 1]])

        gower = GowerMetric(
            np.array(
                [DataType.BINARY_ASYMMETRIC for _ in range(data.shape[1])]
            )
        )
        gower.fit(data)
        res = gower(data[0], data[1])
        self.assertEqual(res, 1 / 5)

    def test_weights(self):
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
        weights = np.array([1, 2, 3, 4, 5])

        dist = np.array([2.45, 0.3, 23, 0.1, 8])
        ranges = np.array([17.04, 18.21, 567, 0.8, 32])

        gower = GowerMetric(
            np.array([DataType.RATIO_SCALE for _ in range(data.shape[1])]),
            weights=weights,
        )
        gower.fit(data)
        res = gower(data[0], data[1])
        self.assertEqual(res, ((dist / ranges) @ weights) / 5.0)


if __name__ == "__main__":
    unittest.main()
