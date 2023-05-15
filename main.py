from Data.data import import_data, pickle_data, features_engineering, analyze_peaks
from utils import get_frequencies, get_frequencies0, extract_line, divide_signal, \
    extract_peaks, normal_distribution, extract_distributions, CSV_DATA, PROJECT_PATH, \
    amplitudes_means_standard_deviations_from_df, amp_means_std_to_peaks
from Preprocessing.tools import LKK, GDRT, L_curve
from Machine_Learning.gradient_descent import gradient_descent
from plot import plot_IS, plot_signal, plot_Residuals, plot_distributions, \
    verify_RC_peaks, plot_3D_signals

import numpy as np
import pandas as pd
import os
import threading


def main():
    original_data = import_data()
    f = get_frequencies()
    f_GDRT = get_frequencies0()
    original_data = original_data[(original_data['CellID'] == 3) &
                                  (original_data['theta'] == 25)]

    # pickle_data(original_data, f, f_GDRT)
    # features_engineering(original_data, 'data_0.1_NOPARASITES.csv')


if __name__ == '__main__':
    main()
