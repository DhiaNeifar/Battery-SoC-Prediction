from Data.data import import_data, pickle_data, features_engineering, CSV_DATA
from utils import get_frequencies, get_frequencies0, extract_line, load_variable
from Preprocessing.tools import LKK, GDRT


import numpy as np


def main():
    data = import_data()
    f = get_frequencies()
    f_GDRT = get_frequencies0()
    # pickle_data(data, f, f_GDRT)
    features_engineering(data)


if __name__ == '__main__':
    main()
