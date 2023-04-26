from plot import plot_distributions, plot_loss
from utils import dump_variable, load_variable


import numpy as np
import random
import matplotlib.pyplot as plt
from tqdm import tqdm


def azer():
    pass


def find_std(means):
    new_stds = np.zeros(means.shape)
    rows = means.shape[0]
    x = (means[1, 0] + means[0, 0]) / 2
    new_stds[0, 0] = x / 3
    i = 2
    while i < rows:
        x1 = (means[i, 0] + means[i - 1, 0]) / 2
        new_stds[i - 1, 0] = min(x, x1) / 3
        x = x1
        i += 1
    new_stds[i - 1, 0] = x / 3
    return new_stds


if __name__ == '__main__':
    _means = np.array([1, 2, 5, 7, 12]).reshape((5, 1))
    print(find_std(_means))

