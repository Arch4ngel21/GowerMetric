import random

from time import sleep

import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm
from scipy.optimize import minimize


def metric(vec_1, vec_2):
    print(f"{vec_1[0]} - {type(vec_1[0])}\n{vec_2} - {type(vec_2)}")
    return np.abs(np.sum(vec_1 - vec_2))


def func(w):
    print("func", w[0] ** 2 + w[1] ** 2)
    return w[0] ** 2 + w[1] ** 2


def derivative(w):
    print("deriv", np.array([2 * w[0] + w[1] ** 2, w[0] ** 2 + 2 * w[1] ** 0]))
    return np.array([2 * w[0] + w[1] ** 2, w[0] ** 2 + 2 * w[1] ** 0])


if __name__ == "__main__":
    with tqdm(total=100, ncols=100) as pbar:
        with tqdm(total=100, ncols=100, position=1) as pbar2:
            for i in range(10):
                sleep(0.3)
                pbar.update(10)
            pbar2.update(50)
