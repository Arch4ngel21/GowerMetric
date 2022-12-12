import random

from time import sleep

import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm
from scipy.optimize import minimize

from enum import Enum

import timeit


class SimpleEnum(Enum):
    example_0 = 0
    example_1 = 1
    example_2 = 2


if __name__ == "__main__":
    # N = 344375
    # vec_1 = np.arange(5)
    # vec_2 = np.arange(5) + 3
    #
    # truth_table = np.array([True for _ in range(5)])
    #
    # start = timeit.default_timer()
    # for _ in range(344375):
    #     vec_3 = vec_1 - vec_2
    #     vec_3 = np.abs(vec_1, vec_2)
    # print(timeit.default_timer() - start)

    a = np.array([1, 2, 3, 4, 5])
    b = np.array([True, False, True, True, True])

    print(a * b)
