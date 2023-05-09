from Data.data import import_data, pickle_data, features_engineering, CSV_DATA, analyze_peaks
from utils import get_frequencies, get_frequencies0, extract_line, load_variable
from plot import plot_3D_signals
from Preprocessing.tools import LKK, GDRT


import numpy as np


def main():
    data = import_data()
    f = get_frequencies()
    f_GDRT = get_frequencies0()

    # pickle_data(data, f, f_GDRT)
    features_engineering(data)
    # cell, theta, cycle = 4, 40, 2
    # analyze_peaks(data, f, f_GDRT, cell, theta, cycle)


if __name__ == '__main__':
    main()
