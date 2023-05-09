from utils import amplitudes_means_from_peaks, CSV_DATA, extract_distributions, \
    normal_distribution


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# TODO : Cleaner + Documented Code

def plot_IS(Z, titles):
    """
    Plot Impedance Spectroscopy
    :param Z: List of Impedance
    :param titles: Name of each Element
    :return:
    """
    for index, z in enumerate(Z):
        plt.plot(z.real, -z.imag, label=f'{titles[index]}')
    plt.title('Impedance Spectroscopy')
    plt.legend()
    plt.axis("equal")
    plt.show()


def plot_signal(signal, freq0, peaks, RC=True):
    # TODO : Draw small circles on the peaks in red for better readability.
    """
    Plot The signal distribution after GDRT Transformation with x indicating peaks of the signal.
    :param peaks:
    :param signal:
    :param freq0:
    :param RC: Whether focus on RC distribution or RL
    :return:
    """
    title = 'Generalized Distribution of Relaxation Times '
    plt.semilogx(np.squeeze(freq0), signal)
    if peaks:
        amplitudes, means = amplitudes_means_from_peaks(peaks)
        plt.plot(means, amplitudes, 'x', color='black', label='peak')
        plt.legend()
    if RC:
        title += 'RC'
    else:
        title += 'RL'
    plt.title(title)
    plt.show()


def plot_Residuals(b, b_hat, residuals):
    """
    Plot the Residual b - b_hat.
    :param b:
    :param b_hat:
    :param residuals:
    :return:
    """
    plt.plot(b, 'o', label='b')
    plt.plot(b_hat, label='b_hat')
    plt.title('b, b_hat')
    plt.legend()
    plt.show()

    plt.plot(residuals)
    plt.title('Residuals')
    plt.show()


def plot_loss(iterations, loss):
    """
    Plot the loss after Gradient Descent Algorithm.
    :param iterations:
    :param loss:
    :return:
    """
    plt.loglog(list(range(iterations)), loss)
    plt.xlabel('Iterations')
    plt.ylabel('Loss')
    plt.show()


def plot_distributions(signal, f_GDRT, normal_dists, show_sum=True):
    """
    Plot different distributions and their sum compared to the original signal.
    :param signal:
    :param f_GDRT:
    :param normal_dists:
    :param show_sum:
    :return:
    """
    ax = np.squeeze(np.exp(f_GDRT))
    for index, distribution in enumerate(normal_dists.T):
        plt.semilogx(ax, distribution, '-', label=f'dist {index + 1}')
    if show_sum:
        plt.semilogx(ax, np.sum(normal_dists, axis=1), '-', label=f'sum')
    plt.semilogx(ax, signal, '-', color='gray', label='signal')
    plt.xlabel('Data points')
    plt.ylabel('Probability Density')
    plt.legend()
    plt.show()


def plot_3D_signals(data, signals, peaks, f_GDRT):
    f_GDRT = np.squeeze(f_GDRT)
    _ = plt.figure(figsize=(10, 20))
    ax = plt.axes(projection='3d')
    SoCs = data['SoC'].unique()
    length = f_GDRT.shape
    for index, signal in enumerate(signals):
        ax.plot3D(np.log(f_GDRT), SoCs[index] * np.ones(length), np.squeeze(signal),
                  label=f'Signal {index + 1} | Peaks = {peaks[index]}')
    ax.legend()
    ax.invert_yaxis()
    ax.set_yticks(SoCs)
    plt.show()


def verify_RC_peaks(signal, line, f_GDRT):
    final_data = pd.read_csv(CSV_DATA(_lambda=0.1))
    amplitudes, means, standard_deviations = extract_distributions(final_data, line)

    signal, f_GDRT = np.squeeze(signal), np.squeeze(f_GDRT)
    f_GDRT = np.log(f_GDRT)
    means = np.log(means)
    normal_dists = normal_distribution(f_GDRT, amplitudes, means, standard_deviations)
    plot_distributions(signal, f_GDRT, normal_dists)
