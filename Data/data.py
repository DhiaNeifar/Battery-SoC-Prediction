from utils import DATA_PATH, get_frequencies, get_frequencies0, \
    extract_line, extract_peaks, divide_signal, dump_variable
from plot import plot_IS, plot_signal, plot_Residuals
from Preprocessing.tools import GDRT, LKK, L_curve
from Machine_Learning.gradient_descent import gradient_descent


from scipy.io import loadmat
import pandas as pd
import numpy as np
import os
from tqdm import tqdm


def Formatting(unprocessed, cycle=True):
    """
    Formats data
    :param unprocessed: Unprocessed data
    :param cycle: Adds a 'Cycle' column for better analysis
    :return: Formatted data as a dataframe for better manipulation
    """
    columns = list(unprocessed.dtype.names)
    if cycle:
        columns.append('Cycle')
    dataframe = pd.DataFrame(columns=columns)
    unprocessed = np.squeeze(unprocessed)
    cycle_index = 1
    cell_ID, theta, SoC, Z_measures, Cycle = [], [], [], [], []

    for line in unprocessed:
        cell, soc = int(line[0]), int(line[2])
        cell_ID.append(cell)
        theta.append(int(line[1]))
        SoC.append(soc)
        Z_measures.append(list(np.squeeze(line[3])))
        data = [cell_ID, theta, SoC, Z_measures]
        if cycle:
            Cycle.append(cycle_index)
            data.append(Cycle)
        dictionary = {k: v for k, v in zip(columns, data)}
        dataframe = pd.DataFrame(dictionary)
        if cell == 4 and soc == 10:
            cycle_index += 1
    return dataframe


def import_data():
    """
    Import Matlab Data
    :return:
    """
    matlab_table = loadmat(DATA_PATH())
    impedance_data = matlab_table["impedanceData2"]
    return Formatting(impedance_data)

    
def save_data(data, freq, freq0):
    lambdas = []
    R, L, C = [], [], []
    RC_means, RC_standard_deviations, RC_Loss = [], [], []
    RL_means, RL_standard_deviations, RL_Loss = [], [], []
    for line in tqdm(range(len(data))):
        Z = extract_line(data, index=line)

        Z = LKK(freq, freq0, Z)

        _lambda = L_curve(freq, freq0, Z)
        lambdas.append(_lambda)

        A, x, b, b_hat, residuals = GDRT(freq, freq0, Z, _lambda=_lambda)
        r, l, c, RC, RL = divide_signal(x)

        R.append(r)
        L.append(l)
        C.append(c)

        RC_peaks = extract_peaks(np.squeeze(RC), np.squeeze(freq0))
        RL_peaks = extract_peaks(np.squeeze(RL), np.squeeze(freq0))

        rc_means, rc_standard_deviations, rc_loss = gradient_descent(np.squeeze(RC), np.squeeze(freq0), RC_peaks)
        el_means, el_standard_deviations, el_loss = gradient_descent(np.squeeze(RL), np.squeeze(freq0), RL_peaks)

        RC_means.append(rc_means)
        RL_means.append(el_means)

        RL_standard_deviations.append(el_standard_deviations)
        RC_standard_deviations.append(rc_standard_deviations)

        RL_Loss.append(el_loss)
        RC_Loss.append(rc_loss)

    dump_variable('Lambdas', lambdas)
    dump_variable('R', R)
    dump_variable('L', L)
    dump_variable('C', C)
    dump_variable('RC_means', RC_means)
    dump_variable('RL_means', RL_means)
    dump_variable('RL_standard_deviations', RL_standard_deviations)
    dump_variable('RC_standard_deviations', RC_standard_deviations)
    dump_variable('RL_Loss', RL_Loss)
    dump_variable('RC_Loss', RC_Loss)


if __name__ == '__main__':
    import_data()
