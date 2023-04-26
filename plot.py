from utils import _peaks


import matplotlib.pyplot as plt
import numpy as np


# TODO : Cleaner + Documented Code

def plot_IS(Z, titles):
    for index, z in enumerate(Z):
        plt.plot(z.real, -z.imag, label=f'{titles[index]}')
    plt.title('Impedance Spectroscopy')
    plt.legend()
    plt.axis("equal")
    plt.show()


def plot_signal(signal, freq0, peaks, RC=True):
    # TODO : Draw small circles on the peaks in red for better readability.
    """
    :param peaks:
    :param signal:
    :param freq0:
    :param RC: Whether focus on RC distribution or RL
    :return:
    """
    title = 'Generalized Distribution of Relaxation Times '
    plt.semilogx(np.squeeze(freq0), signal)
    if peaks:
        amplitudes, means = _peaks(peaks)
        plt.plot(means, amplitudes, 'x', color='black', label='peak')
        plt.legend()
    if RC:
        title += 'RC'
    else:
        title += 'RL'
    plt.title(title)
    plt.show()


def plot_Residuals(b, b_hat, residuals):
    plt.plot(b, 'o', label='b')
    plt.plot(b_hat, label='b_hat')
    plt.title('b, b_hat')
    plt.legend()
    plt.show()

    plt.plot(residuals)
    plt.title('Residuals')
    plt.show()


def plot_loss(iterations, loss):
    plt.loglog(list(range(iterations)), loss)
    plt.xlabel('Iterations')
    plt.ylabel('Loss')
    plt.show()


def plot_distributions(signal, x_axis, normal_dists, show_sum=True):
    ax = np.squeeze(x_axis)
    for index, distribution in enumerate(normal_dists.T):
        plt.semilogx(ax, distribution, '-', label=f'dist {index + 1}')
    if show_sum:
        plt.semilogx(ax, np.sum(normal_dists, axis=1), '-', label=f'sum')
    plt.semilogx(ax, signal, '-', color='gray', label='signal')
    plt.xlabel('Data points')
    plt.ylabel('Probability Density')
    plt.legend()
    plt.show()
