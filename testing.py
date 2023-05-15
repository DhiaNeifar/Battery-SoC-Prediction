from Data.data import import_data, pickle_data, features_engineering, CSV_DATA, \
    analyze_peaks
from utils import get_frequencies, get_frequencies0, extract_line, load_variable, \
    PROJECT_PATH, divide_signal, extract_peaks, amplitudes_means_standard_deviations_from_df, \
    normal_distribution, amp_means_std_to_peaks
from plot import plot_3D_signals, plot_signal, plot_distributions
from Preprocessing.tools import LKK, GDRT
from Machine_Learning.gradient_descent import gradient_descent

import numpy as np
import os
import pandas as pd


def test():
    data = import_data()
    f = get_frequencies()
    f_GDRT = get_frequencies0()
    cell, theta = 3, 25
    data = data[(data['CellID'] == cell) & (data['theta'] == theta)]
    line = 169
    Z = extract_line(data, index=line)

    Z = LKK(f, f_GDRT, Z)

    r = np.min(Z.real)
    A, x, b, b_hat, residuals = GDRT(f, f_GDRT, Z - r, _lambda=0.1)
    l, c, RC, RL = divide_signal(x)
    RC_peaks = extract_peaks(RC, f_GDRT)
    rc_amplitudes, rc_means, rc_standard_deviations, rc_loss = gradient_descent(RC, f_GDRT, RC_peaks,
                                                                                parameters=['s'],
                                                                                peak=None, search_std=True,
                                                                                plot=True)


def fitting_peak3_cell3_temp25():
    PATH = os.path.join(PROJECT_PATH(), 'Data', 'data_0.1_CELL3_TEMP25_peak_1.csv_peak_1_3.csv')
    data = pd.read_csv(PATH, index_col=0)
    original_data = import_data()
    for cycle in range(1, 23):
        print(f'cycle {cycle}/23')
        line = data[(data['SoC'] == 10) & (data['Cycle'] == cycle)]
        amplitudes, means, standard_deviations = amplitudes_means_standard_deviations_from_df(line)

        index = int(original_data[(original_data['theta'] == 25) &
                                  (original_data['CellID'] == 3) &
                                  (original_data['SoC'] == 10) &
                                  (original_data['Cycle'] == cycle)].index[0])

        f = get_frequencies()
        f_GDRT = get_frequencies0()
        _Z = extract_line(original_data, index=index)

        Z = LKK(f, f_GDRT, _Z)
        r = np.min(Z.real)
        _, x, _, _, _ = GDRT(f, f_GDRT, Z - r, _lambda=0.1, regularized=True)
        _, _, RC, _ = divide_signal(x)
        new_peaks = amp_means_std_to_peaks(amplitudes, means, standard_deviations)
        peaks = [3]
        print('before')
        print(amplitudes)
        print(means)
        print(standard_deviations)

        amplitudes, means, standard_deviations, _ = gradient_descent(RC, f_GDRT, new_peaks, parameters=['a', 's', 'm'],
                                                                     peak=peaks, search_std=False, plot=False)
        print('after')
        print(amplitudes)
        print(means)
        print(standard_deviations)

        for peak in peaks:
            data.loc[(data['SoC'] == 10) & (data['Cycle'] == cycle), f'RC_amplitude_{peak}'] = amplitudes[peak - 1, 0]
            data.loc[(data['SoC'] == 10) & (data['Cycle'] == cycle), f'RC_mean_{peak}'] = means[peak - 1, 0]
            data.loc[(data['SoC'] == 10) & (data['Cycle'] == cycle), f'RC_standard_deviation_{peak}'] = standard_deviations[peak - 1, 0]

    data.to_csv('data_0.1_PostProcessing_Cell3_Temp25_peak_1_30.csv', index=True)


if __name__ == '__main__':
    main()
