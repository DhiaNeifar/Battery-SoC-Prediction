from plot import plot_distributions, plot_loss
from utils import extract_peaks, amplitudes_means_std_from_peaks, find_std, \
    normal_distribution


import numpy as np
import random
import matplotlib.pyplot as plt
from tqdm import tqdm


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


def Compute_Gradients(signal, axis, amplitudes, means, stds, distributions):
    """
    Updates the parameters means & standard-deviations
    :param signal:          shape == (m,)
    :param axis:            shape == (m,)
    :param amplitudes:      shape == (num_peaks, 1)
    :param means:           shape == (num_peaks, 1)
    :param stds:            shape == (num_peaks, 1)
    :param distributions:   shape == (num_peaks, m)
    :return: Updated parameters
    """
    m = signal.shape[0]
    cost = np.sum(distributions, axis=1) - signal
    x_mean = (axis - means).T
    stds = stds.T
    squared_stds = np.square(stds)

    new_amplitudes = np.expand_dims(np.sum(distributions * np.expand_dims(cost, axis=1) / m, axis=0), axis=1)

    new_means = amplitudes * np.expand_dims(np.sum(x_mean * distributions * np.expand_dims(cost, axis=1) / (m * squared_stds), axis=0), axis=1)

    cubed_stds = squared_stds * stds
    new_stds = amplitudes * np.expand_dims(np.sum(np.square(x_mean) * distributions * np.expand_dims(cost, axis=1) / (m * cubed_stds), axis=0), axis=1)

    return new_amplitudes, new_means, new_stds


def gradient_descent(signal, f_GDRT, peaks, max_iter=100000, lr_amp=1e3, lr_mean=1e8, lr_std=1e8, parameters=None,
                     peak=None, search_std=True, plot=True):
    # TODO : Add better visualisation method
    """
    Gradient Descent Algorithm.
    :param signal: Original signal to fit
    :param f_GDRT: f_GDRT (x Axis) of the signal
    :param peaks: Peaks of the signal
    :param max_iter: Number of iteration till convergence of the algorithm
    :param lr_amp: Learning rate for the amplitudes
    :param lr_mean: Learning rate for the means
    :param lr_std: Learning rate for the standard deviations
    :param peak: Which peak to update
    :param parameters: Which parameter to update
    :param search_std: If you want to search for approximate standard_deviations
    :param plot: If you want to plot distributions and have visualization
    :return: If peaks empty return Null if peaks empty return [0, 0]
    """
    def update(params, gradient, lr, _elements):
        if _elements == 'all':
            return params - lr * gradient
        new_params = np.copy(params)
        for _element in _elements:
            new_params[_element - 1, 0] = new_params[_element - 1, 0] - lr * gradient[_element - 1, 0]
        return new_params

    if peak is None:
        peak = 'all'
    if parameters is None:
        parameters = ['a', 'm', 's']

    signal, f_GDRT = np.squeeze(signal), np.squeeze(f_GDRT)

    if not peaks:
        null = np.zeros((1, 1))
        return null, null, null, [-1]

    amplitudes, means, standard_deviations = amplitudes_means_std_from_peaks(peaks)

    f_GDRT = np.log(f_GDRT)
    means = np.log(means)

    if search_std:
        standard_deviations = find_std(means)

    normal_dists = normal_distribution(f_GDRT, amplitudes, means, standard_deviations)
    if plot:
        plot_distributions(signal, f_GDRT, normal_dists, show_sum=False)
    J = ComputeCost(signal, normal_dists)

    Loss = [J]
    for _ in tqdm(range(max_iter)):
        new_amplitudes, new_means, new_std_divs = Compute_Gradients(signal, f_GDRT, amplitudes, means, standard_deviations, normal_dists)

        if 'a' in parameters:
            amplitudes = update(amplitudes, new_amplitudes, lr_amp, peak)
        if 'm' in parameters:
            means = update(means, new_means, lr_mean, peak)
        if 's' in parameters:
            standard_deviations = update(standard_deviations, new_std_divs, lr_std, peak)

        normal_dists = normal_distribution(f_GDRT, amplitudes, means, standard_deviations)

        J = ComputeCost(signal, normal_dists)
        Loss.append(J)
    normal_dists = normal_distribution(f_GDRT, amplitudes, means, standard_deviations)
    if plot:
        plot_loss(max_iter + 1, Loss)
        plot_distributions(signal, f_GDRT, normal_dists)
    return amplitudes, np.exp(means), np.abs(standard_deviations), Loss


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
