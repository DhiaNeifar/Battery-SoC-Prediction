from plot import plot_distributions, plot_loss
from utils import extract_peaks, _peaks


import numpy as np
import random
import matplotlib.pyplot as plt
from tqdm import tqdm


def normal_distribution(x_axis, amplitude, mean, std):
    return (amplitude * np.exp(-0.5 * ((x_axis - mean) / std) ** 2)).T


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


def ComputeCost(sig, dists):
    """
    Computes the Cost J
    :param sig:     shape == (m,)
    :param dists:   shape == (m, num_peaks)
    :return: J : Cost
    """
    m = sig.shape[0]
    cost = np.sum(dists, axis=1) - sig
    return cost.T @ cost / (2 * m)


def Compute_Gradients(signal, axis, means, stds, distributions):
    """
    Updates the parameters means & standard-deviations
    :param stds:
    :param means:           shape == (num_peaks, 1)
    :param axis:            shape == (m,)
    :param signal:          shape == (m,)
    :param distributions:   shape == (num_peaks, m)
    :return: means : new updated means
    """
    m = signal.shape[0]
    cost = np.sum(distributions, axis=1) - signal
    x_mean = (axis - means).T
    stds = stds.T
    squared_stds = np.square(stds)
    new_means = np.sum(x_mean * distributions * np.expand_dims(cost, axis=1) / (m * squared_stds), axis=0)

    cubed_stds = squared_stds * stds
    new_stds = np.sum(np.square(x_mean) * distributions * np.expand_dims(cost, axis=1) / (m * cubed_stds), axis=0)
    return np.expand_dims(new_means, axis=1), np.expand_dims(new_stds, axis=1)


def gradient_descent(signal, x_axis, peaks, max_iter=100000, lr_std=100000):
    # TODO Add better visualisation method
    """
    Gradient Descent Algorithm
    :param signal: Original signal to fit
    :param x_axis: x Axis of the signal
    :param peaks: Peaks of the signal
    :param max_iter: Number of iteration till convergence of the algorithm
    :param lr_std: Learning rate for the standard deviations
    :return: if peaks empty return Null if peaks empty return [0, 0]
    """

    if not peaks:
        null = np.ones((1, 1))
        return null, null, []

    amplitudes, means = _peaks(peaks)
    _means = np.copy(means)

    x_axis = np.log(x_axis)
    means = np.log(means)

    standard_deviations = find_std(means)

    normal_dists = normal_distribution(x_axis, amplitudes, means, standard_deviations)
    # plot_distributions(signal, np.exp(x_axis), normal_dists, show_sum=False)
    J = ComputeCost(signal, normal_dists)

    Loss = [J]
    for _ in range(max_iter):
        new_means, new_std_divs = Compute_Gradients(signal, x_axis, means, standard_deviations, normal_dists)

        # means -= lr_means * new_means
        standard_deviations -= lr_std * new_std_divs

        normal_dists = normal_distribution(x_axis, amplitudes, means, standard_deviations)
        J = ComputeCost(signal, normal_dists)

        Loss.append(J)
    # plot_loss(iteration + 1, Loss)
    normal_dists = normal_distribution(x_axis, amplitudes, means, standard_deviations)
    # plot_distributions(signal, np.exp(x_axis), normal_dists)
    return _means, np.abs(standard_deviations), Loss


def main():
    # Endgame initialization
    m = 10000
    x_axis = np.logspace(1, 2, m)

    real_amplitudes = np.array([100, 50]).reshape((2, 1))
    real_means = np.array([50, 30]).reshape((2, 1))
    real_std = np.array([4, 5]).reshape((2, 1))

    y1 = normal_distribution(x_axis, real_amplitudes[0, 0], real_means[0, 0], real_std[0, 0])
    y2 = normal_distribution(x_axis, real_amplitudes[1, 0], real_means[1, 0], real_std[1, 0])
    signal = y1 + y2

    peaks = extract_peaks(signal, x_axis)

    normal_dists = normal_distribution(x_axis, amplitudes, means, std)
    plot_distributions(signal, x_axis, normal_dists)

    gradient_descent(signal, x_axis, peaks)


if __name__ == '__main__':
    main()
