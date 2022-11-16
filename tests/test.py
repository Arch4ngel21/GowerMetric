import unittest

import numpy as np
from sklearn.preprocessing import OrdinalEncoder

from PoC import GowerMetric, DataType, bin_dist


class SingleDataTypeTests(unittest.TestCase):
    def test_cat_nom(self):

        data = np.array(
            [
                ["male", "Poland", "Senior developer", "married", "SysOps"],
                ["female", "Poland", "Junior developer", "married", "DevOps"],
                ["male", "England", "Senior developer", "unmarried", "DevOps"],
            ]
        )

        data_cols_type = np.array(
            [DataType.CATEGORICAL_NOMINAL for _ in range(len(data[0]))]
        )

        enc = OrdinalEncoder()
        enc.fit(data)

        encoded_data = enc.transform(data)

        gower = GowerMetric(data_cols_type, "iqr")
        gower.fit(encoded_data)

        self.assertTrue(
            np.isclose(gower(encoded_data[0], encoded_data[1]), 0.6)
        )

    def test_ratio(self):
        data = np.array(
            [
                [0.0, 15.0, 50.0, 100.0, 80.0],
                [0.0, 25.0, 70.0, 100.0, 80.0],
                [1.0, 30.0, 80.0, 5.0, 70.0],
                [1.0, 20.0, 40.0, 25.0, 45.0],
            ]
        )

        data_cols_type = np.array(
            [DataType.RATIO_SCALE for _ in range(len(data[0]))]
        )

        gower = GowerMetric(data_cols_type, "iqr")
        gower.fit(data)

        self.assertTrue(
            np.isclose(gower(data[0], data[1]), (1.0 + 20.0 / 25.0) / 5)
        )

    def test_mixed_data(self):
        data = np.array(
            [
                ["M", 18, "Poland", "Senior developer", "PhD", 98.0],
                ["F", 20, "Finland", "Senior developer", "MSc", 92.0],
                ["M", 20, "England", "Junior developer", "MSc", 72.0],
                ["F", 30, "Poland", "Senior developer", "PhD", 80.0],
                ["F", 17, "Poland", None, None, 54.0],
            ]
        )

        data_cols_type = np.array(
            [
                DataType.CATEGORICAL_NOMINAL,
                DataType.RATIO_SCALE,
                DataType.CATEGORICAL_NOMINAL,
                DataType.CATEGORICAL_NOMINAL,
                DataType.CATEGORICAL_NOMINAL,
                DataType.RATIO_SCALE,
            ]
        )

        enc = OrdinalEncoder()
        enc.fit(data[:, [0, 2, 3, 4]])
        data[:, [0, 2, 3, 4]] = enc.transform(data[:, [0, 2, 3, 4]])

        gower = GowerMetric(data_cols_type)
        gower.fit(data)

        self.assertTrue(
            np.isclose(gower(data[0], data[1]), (1.0 + 1.0 + 1.0) / 6)
        )
        self.assertTrue(
            np.isclose(gower(data[0], data[4]), (1.0 + 1.0 + 1.0 + 1.0) / 6)
        )
        self.assertTrue(
            np.isclose(
                gower(data[0], data[3]), (1.0 + 12.0 / 13.0 + 18.0 / 44.0) / 6
            )
        )


class GowerVsBinary(unittest.TestCase):
    def test_mixed_data(self):
        data = np.array(
            [
                ["M", 18, "Poland", "Senior developer", "PhD", 98.0],
                ["F", 20, "Finland", "Senior developer", "MSc", 92.0],
                ["M", 20, "England", "Junior developer", "MSc", 72.0],
                ["F", 30, "Poland", "Senior developer", "PhD", 80.0],
                ["F", 17, "Poland", None, None, 54.0],
            ]
        )

        data_cols_type = np.array(
            [
                DataType.CATEGORICAL_NOMINAL,
                DataType.RATIO_SCALE,
                DataType.CATEGORICAL_NOMINAL,
                DataType.CATEGORICAL_NOMINAL,
                DataType.CATEGORICAL_NOMINAL,
                DataType.RATIO_SCALE,
            ]
        )

        enc = OrdinalEncoder()
        enc.fit(data[:, [0, 2, 3, 4]])
        data[:, [0, 2, 3, 4]] = enc.transform(data[:, [0, 2, 3, 4]])
        data = data.astype(dtype=np.float64)

        gower = GowerMetric(data_cols_type)
        gower.fit(data)

        self.assertLessEqual(
            gower(data[0], data[2]), bin_dist(data[0], data[2])
        )
        self.assertLessEqual(
            gower(data[1], data[2]), bin_dist(data[1], data[2])
        )


if __name__ == "__main__":
    unittest.main()
