from Data.data import import_data, pickle_data
from utils import get_frequencies, get_frequencies0, extract_line, divide_signal, \
    extract_peaks
from Preprocessing.tools import LKK, GDRT, L_curve
from Machine_Learning.gradient_descent import gradient_descent
from plot import plot_IS, plot_signal, plot_Residuals


import numpy as np


def main():
    data = import_data()
    f = get_frequencies()
    f_GDRT = get_frequencies0()
    line = 851
    _Z = extract_line(data, index=line)
    Z = LKK(f, f_GDRT, _Z)
    # plot_IS([_Z, Z], ['Original Signal', 'After LKK Transformation'])
    r = np.min(Z.real)
    # _lambda = L_curve(f, f_GDRT, Z)
    # print(_lambda)
    A, x, b, b_hat, residuals = GDRT(f, f_GDRT, Z - r, _lambda=0.1, regularized=True)
    # plot_Residuals(b, b_hat, residuals)
    L, C, RC, RL = divide_signal(x)
    RC_peaks = extract_peaks(RC, f_GDRT)
    plot_signal(RC, f_GDRT, RC_peaks, RC=True)
    means, standard_deviations, J = gradient_descent(RC, f_GDRT, RC_peaks, max_iter=100000, lr_std=100000, plot=True)


if __name__ == '__main__':
    main()
