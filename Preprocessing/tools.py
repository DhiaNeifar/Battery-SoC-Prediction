from utils import extract_peaks

from scipy.optimize import nnls
from numpy.linalg import lstsq
import os
import numpy as np
from math import pi
import matplotlib.pyplot as plt


def LKK(freq, freq0, Z):
    """
    Apply LKK transformation to the EIS data.
    :param freq:
    :param freq0:
    :param Z:
    :return: LKK-transformed EIS data.
    """
    coefficients = 1 / (1 + 1j * freq / freq0)
    A = np.vstack((coefficients.real, coefficients.imag))
    b = np.vstack((Z.real, Z.imag))
    x = lstsq(A, np.squeeze(b), rcond=None)[0]
    b_hat = A @ x
    new_Z = b_hat[: len(Z)] + 1j * b_hat[len(Z):]
    return new_Z.reshape((-1, 1))


def GDRT(freq, freq0, Z, _lambda=0.1, regularized=True):
    """
    Generalized Distribution of Relaxation Function
    :param freq:
    :param freq0:
    :param Z:
    :param _lambda: Regularization Parameter
    :param regularized:
    :return:
        m : f shape
        n : f0 shape
        A : GDRT Coefficients   shape ==> (2(m + n) + 3, 2n + 3)
        x : Solutions           shape ==> (2n + 3, 1)
        b : [Re(Z), Im(Z)]      shape ==> (2(m + n) + 3, 1)
        b_hat : A @ x           shape ==> (2(m + n) + 3, 1)
        residuals : b - b_hat   shape ==> (2(m + n) + 3, 1)
    """

    m = freq.shape[0]
    n = freq0.shape[1]
    R0 = np.min(Z.real)

    def MDL_RC(_freq, _freq0):
        return 1 / (1 + 1j * _freq / _freq0)

    def MDL_RL(_freq, _freq0):
        W = _freq / _freq0
        return (1j * W) / (1 + 1j * W)

    def calculate_A(_freq, _freq0):
        A_RC, A_RL = MDL_RC(_freq, _freq0), MDL_RL(_freq, _freq0)
        _A = np.concatenate(
            (np.concatenate((A_RC.real, A_RL.real), axis=1), np.concatenate((A_RC.imag, A_RL.imag), axis=1)), axis=0)
        RR = np.ones((m, 1))
        X1 = np.zeros((m, 2))
        X2 = np.zeros((m, 1))
        LL = 2 * pi * freq
        CC = -1 / (2 * pi * freq)

        TMP = np.concatenate((np.concatenate((RR, X1), axis=1), np.concatenate((X2, LL, CC), axis=1)), axis=0)

        return np.concatenate((TMP, _A), axis=1)

    def calculate_b(impedance):
        s = 2 * n + 3
        return np.concatenate((impedance.real, impedance.imag, np.zeros((s, 1))), axis=0)

    A = calculate_A(freq, freq0)
    b = calculate_b(Z - R0)

    if regularized:
        A = np.concatenate((A, _lambda * np.eye(A.shape[-1])), axis=0)

    x = nnls(A, np.squeeze(b), maxiter=10 ** 9)[0]
    x = np.expand_dims(x, axis=1)

    b_hat = A @ x
    residuals = b - b_hat

    return A, x, b, b_hat, residuals


def L_curve(freq, freq0, Z):
    # TODO Add plot function for L_curve
    """
    The use of L-Curve technique to find the most optimum regularization Term.
    :param freq:
    :param freq0:
    :param Z:
    :return: return Best Lambda.
    """

    def get_lambdas(n=4):
        num1, num2 = 3, 1
        jump = 10
        l = []
        for i in range(n):
            l.append(num1)
            l.append(num2)
            num1 /= jump
            num2 /= jump
        return l

    def extract_lambda(_loss, _lambdas):
        slope, diff = [], []
        i = 1
        slope.append((_loss[i] - _loss[i - 1]) / (_lambdas[i] - _lambdas[i - 1]))
        for i in range(2, len(_loss)):
            slope.append((_loss[i] - _loss[i - 1]) / (_lambdas[i] - _lambdas[i - 1]))
            diff.append(abs(slope[i - 1] - slope[i - 2]))
        return _lambdas[np.argmax(np.array(diff)) + 2]

    lambdas = get_lambdas()
    loss = []

    for _lambda in lambdas:
        _, x, _, _, residuals = GDRT(freq, freq0, Z, _lambda=_lambda, regularized=True)
        loss.append((residuals.T @ residuals)[0, 0])
    # fig, ax1 = plt.subplots()
    # ax1.set_xlabel('lambda')
    # ax1.set_ylabel('Loss')
    # ax1.semilogx(lambdas, loss, 'x-')
    # ax1.invert_xaxis()
    # ax1.tick_params(axis='y')
#
    # fig.tight_layout()
    # plt.show()
    return extract_lambda(loss, lambdas)


def test():
    pass


if __name__ == '__main__':
    test()
