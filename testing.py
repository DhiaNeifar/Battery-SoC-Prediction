from Data.data import import_data, pickle_data, features_engineering, CSV_DATA, analyze_peaks
from utils import get_frequencies, get_frequencies0, extract_line, load_variable
from plot import plot_3D_signals
from Preprocessing.tools import LKK, GDRT


import numpy as np
import pandas as pd


def main():
    data = import_data()
    f = get_frequencies()
    f_GDRT = get_frequencies0()
    final_data = pd.read_csv(CSV_DATA(_lambda=0.1))



if __name__ == '__main__':
    main()
