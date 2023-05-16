import numpy

from utils import PROJECT_PATH, DATA_PATH, extract_line, divide_signal, extract_peaks, \
    dump_variable, load_variable, find_max, PICKLED_DATA_PATH, unprocessed_pickled_data, \
    CSV_DATA, amplitudes_means_std_from_peaks, get_frequencies, get_frequencies0, \
    amplitudes_means_standard_deviations_from_df, normal_distribution
from Preprocessing.tools import LKK, L_curve, GDRT
from Machine_Learning.gradient_descent import gradient_descent
from plot import plot_3D_signals, plot_distributions, plot_signal, plot_IS

from scipy.io import loadmat
import pandas as pd
import numpy as np
import os
import time
import threading


def Formatting(unprocessed, cycle=True):
    """
    Formats data.
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
    initial_theta = int(unprocessed[0][1])
    for line in unprocessed:
        cell, _theta, soc = int(line[0]), int(line[1]), int(line[2])
        cell_ID.append(cell)
        theta.append(_theta)
        SoC.append(soc)
        Z_measures.append(list(np.squeeze(line[3])))
        data = [cell_ID, theta, SoC, Z_measures]
        if _theta != initial_theta:
            cycle_index = 1
            initial_theta = _theta
        if cycle:
            Cycle.append(cycle_index)
            data.append(Cycle)
        if cell == 4 and soc == 10:
            cycle_index += 1
        dictionary = {k: v for k, v in zip(columns, data)}
        dataframe = pd.DataFrame(dictionary)
    return dataframe


def import_data():
    """
    Import Matlab Data
    :return:
    """
    matlab_table = loadmat(DATA_PATH())
    impedance_data = matlab_table["impedanceData2"]
    return Formatting(impedance_data)


def pickle_data(data, f, f_GDRT):
    """
    Pickle the data received after iterating the whole dataset.
    :param data:
    :param f:
    :param f_GDRT:
    :return:
    """
    parasites = False
    R, L, C = [], [], []
    RC_peaks_length, RL_peaks_length = [], []
    RC_means, RC_standard_deviations, RC_amplitudes, RC_Loss = [], [], [], []
    RL_means, RL_standard_deviations, RL_amplitudes, RL_Loss = [], [], [], []
    length = len(data)
    for index, line in enumerate(range(length), 1):
        print(f'{index}/{length}')
        Z = extract_line(data, index=line)
        print(Z)
        Z = LKK(f, f_GDRT, Z)

        r = np.min(Z.real)
        A, x, b, b_hat, residuals = GDRT(f, f_GDRT, Z - r, _lambda=0.1, parasites=parasites)
        l, c, RC, RL = divide_signal(x, parasites=parasites)

        R.append(r)
        L.append(l)
        C.append(c)

        RC_peaks = extract_peaks(RC, f_GDRT)
        RC_peaks_length.append(len(RC_peaks))

        rc_amplitudes, rc_means, rc_standard_deviations, rc_loss = gradient_descent(RC, f_GDRT, RC_peaks,
                                                                                    parameters=['s'],
                                                                                    peak=None, search_std=True,
                                                                                    plot=False)
        RC_means.append(rc_means)
        RC_standard_deviations.append(rc_standard_deviations)
        RC_amplitudes.append(rc_amplitudes)
        RC_Loss.append(rc_loss[-1])

        RL_peaks = extract_peaks(RL, f_GDRT)
        RL_peaks_length.append(len(RL_peaks))

        rl_amplitudes, rl_means, rl_standard_deviations, rl_loss = gradient_descent(RL, f_GDRT, RL_peaks,
                                                                                    parameters=['s'],
                                                                                    peak=None, search_std=True,
                                                                                    plot=False)
        RL_means.append(rl_means)
        RL_standard_deviations.append(rl_standard_deviations)
        RL_amplitudes.append(rl_amplitudes)
        RL_Loss.append(rl_loss[-1])

    dump_variable('R', R)
    dump_variable('L', L)
    dump_variable('C', C)

    dump_variable('RC_means', RC_means)
    dump_variable('RL_means', RL_means)

    dump_variable('RC_standard_deviations', RC_standard_deviations)
    dump_variable('RL_standard_deviations', RL_standard_deviations)

    dump_variable('RC_amplitudes', RC_amplitudes)
    dump_variable('RL_amplitudes', RL_amplitudes)

    dump_variable('RC_Loss', RC_Loss)
    dump_variable('RL_Loss', RL_Loss)

    dump_variable('RC_peaks', RC_peaks_length)
    dump_variable('RL_peaks', RL_peaks_length)


def features_engineering(data, filename):
    """
    Feature engineering the data and saving it in CSV format.
    Create dataframe with features to use for Machine Learning
    :param filename: filename of saved data as csv
    :param data:
    :return:
    """

    def organize_columns(files):
        def reorder_columns(_list) -> list:
            indexes = [0, 3]
            new_list = _list.copy()
            for _index in indexes:
                element = _list[_index]
                new_list.remove(element)
                new_list.append(element)
            return new_list

        def create_columns(_COLUMNS):
            peaks = load_variable(_COLUMNS[-1])
            _maximum = max(peaks)
            new_columns = []
            for column in _COLUMNS[:-2]:
                new_columns.extend([column[:-1] + f'_{_index}' for _index in range(1, _maximum + 1)])
            new_columns.sort(key=lambda col: int(col.split('_')[-1]))
            return new_columns + _COLUMNS[-2:]

        files.sort(reverse=True)
        PARASITES = files[-3:]
        files = files[:-3]
        files.sort(key=lambda file: file.split('_')[1])
        files.sort(key=lambda file: file.split('_')[0])
        RC_columns, RL_columns = files[:len(files) // 2], files[len(files) // 2:]
        RC_columns, RL_columns = reorder_columns(RC_columns), reorder_columns(RL_columns)
        RC_columns, RL_columns = create_columns(RC_columns), create_columns(RL_columns)
        return [*PARASITES, *RC_columns, *RL_columns]

    pickle_files = unprocessed_pickled_data()
    COLUMNS = data.columns
    global pickled_variable
    global _type
    for pickle_file in pickle_files:
        try:
            pickled_variable = load_variable(pickle_file)
        except EOFError:
            print(f'Error Loading Variable {pickle_file}')
        if pickled_variable:
            _type = type(pickled_variable[0])
            if _type is int or _type is numpy.float64:
                data.insert(len(data.columns), pickle_file, pickled_variable, True)
            if _type is numpy.ndarray:
                maximum = find_max(pickled_variable)
                lists, columns = [], []
                for index in range(maximum):
                    lists.append([])
                    columns.append(f'{pickle_file[:-1]}_{index + 1}')
                for var in pickled_variable:
                    x = var.shape[0]
                    for i in range(x):
                        lists[i].append(var[i, 0])
                    for i in range(x, maximum):
                        lists[i].append(0)
                for i in range(maximum):
                    data.insert(len(data.columns), columns[i], lists[i], True)
    data = data[[*COLUMNS, *organize_columns(pickle_files)]]
    data.to_csv(CSV_DATA(filename), index=True)


def analyze_peaks(data, f, f_GDRT, cell, theta, cycle):
    data = data[(data['CellID'] == cell) & (data['theta'] == theta) & (data['Cycle'] == cycle)]
    RC_signals, RL_signals = [], []
    parasites = False
    for line in range(len(data)):
        _Z = extract_line(data, index=line)
        Z = LKK(f, f_GDRT, _Z)
        r = np.min(Z.real)
        _, x, _, _, _ = GDRT(f, f_GDRT, Z - r, _lambda=0.1, regularized=True, parasites=parasites)
        _, _, RC, RL = divide_signal(x, parasites=parasites)
        RC_signals.append(RC)
        RL_signals.append(RL)
    plot_3D_signals(data, RC_signals, f_GDRT)
    plot_3D_signals(data, RL_signals, f_GDRT)


def postprocessing():
    data = import_data()
    f = get_frequencies()
    f_GDRT = get_frequencies0()
    line = 586
    _Z = extract_line(data, index=line)
    Z = LKK(f, f_GDRT, _Z)
    plot_IS([_Z, Z], ['Original Signal', 'After LKK Transformation'])
    r = np.min(Z.real)
    # _lambda = L_curve(f, f_GDRT, Z)
    # print(_lambda)


def split_data(theta = 25):
    data_file = 'data_0.1.csv'
    data_path = os.path.join(PROJECT_PATH(), 'Data')
    df = pd.read_csv(os.path.join(data_path, data_file))
    df = df[df['theta'] == theta]
    cells = [i for i in range(1, 5)]
    for cell in cells:
        dataframe = df[df['CellID'] == cell]
        dataframe_name = ''.join([data_file[:-4], f'_CELL{cell}', f'_TEMP{theta}_', '.csv'])

        dataframe.to_csv(os.path.join(data_path, dataframe_name), index=True)


def data_verification():
    cell, theta, soc, cycle = 3, 25, 60, 16

    original_data = import_data()

    data_file = '/home/dhianeifar/PFE/3/data_0.1_PostProcessing_Cell3_Temp25_peak_1_30_5.csv'
    new_data = pd.read_csv(os.path.join(PROJECT_PATH(), data_file), index_col=0)
    new_data = new_data[(new_data['CellID'] == cell) &
                        (new_data['theta'] == theta) &
                        (new_data['SoC'] == soc) &
                        (new_data['Cycle'] == cycle)]
    index = new_data.index.to_list()[0]
    f = get_frequencies()
    f_GDRT = get_frequencies0()

    _Z = extract_line(original_data, index=index)
    Z = LKK(f, f_GDRT, _Z)
    r = np.min(Z.real)
    _, x, _, _, _ = GDRT(f, f_GDRT, Z - r, _lambda=0.1, regularized=True)
    _, _, RC, _ = divide_signal(x)
    amplitudes, means, standard_deviations = amplitudes_means_standard_deviations_from_df(new_data)
    normal_dists = normal_distribution(np.log(np.squeeze(f_GDRT)), amplitudes, np.log(means), standard_deviations)
    plot_distributions(np.squeeze(RC), np.log(np.squeeze(f_GDRT)), normal_dists)


if __name__ == '__main__':
    split_data()
