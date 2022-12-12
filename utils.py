from dataclasses import dataclass
from enum import Enum


class DataType(Enum):
    BINARY_SYMMETRIC = 0
    BINARY_ASYMMETRIC = 1
    CATEGORICAL_NOMINAL = 2
    RATIO_SCALE = 3


@dataclass
class Dataset:
    name: str
    task: str
    metric: str
    labeled: bool


@dataclass
class Data:
    cols_type_maping = {
        "bin_sym": DataType.BINARY_SYMMETRIC,
        "cat_nom": DataType.CATEGORICAL_NOMINAL,
        "ratio": DataType.RATIO_SCALE,
        "NULL": None,
    }
    data: dict
    cols_type: dict
    labels: dict
