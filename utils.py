import os
import numpy as np
import pickle


def PROJECT_PATH():
    main_folder = 'Battery-SoC-Prediction'
    path = os.path.join(os.getcwd())
    i, j = len(path) - 1, len(main_folder) - 1
    while i != j:
        k = j
        index_i = i
        while k > -1:
            if main_folder[k] != path[index_i]:
                break
            k -= 1
            index_i -= 1
        if k == -1:
            return path[: i + 1]
        i -= 1


def DATA_PATH():
    return os.path.join(PROJECT_PATH(), 'Data', 'Data_20170726_ICR18650_30A.mat')


def dump_frequencies():
    f = np.array([0.01000, 0.03667, 0.05000, 0.05667, 0.06333, 0.07000, 0.08000, 0.09000, 0.10000, 0.11333, 0.12667, 0.14000, 0.16000, 0.17667, 0.20000, 0.22333, 0.25000, 0.28333, 0.31667, 0.35333, 0.39667, 0.44667, 0.50000, 0.56333, 0.63000, 0.70667, 0.79333, 0.89000, 1.00000, 1.12333, 1.26000, 1.41333, 1.58333, 1.77667, 1.99667, 2.24000, 2.51333, 2.82000, 3.16333, 3.54667, 3.98000, 4.46667, 5.01333, 5.62333, 6.31000, 7.08000, 7.94333, 8.91333, 10.00000, 11.22000, 12.59000, 14.12667, 15.85000, 17.78333, 19.95333, 22.38667, 25.12000, 28.18333, 31.62333, 35.48000, 39.81000, 44.66667, 50.12000, 56.23333, 63.09667, 70.79333, 79.43333, 89.12333, 100.00000, 112.20333, 125.89333, 141.25333, 158.49000, 177.82667, 199.52667, 223.87333, 251.18667, 281.83667, 316.22667, 354.81333, 398.10667, 446.68333, 501.18667, 562.34000, 630.95667, 707.94333, 794.32667, 891.25000, 1000.00000])
    print(f.shape)
    with open('Pickle data/frequencies.pickle', 'wb') as handle:
        pickle.dump(f, handle, protocol=pickle.HIGHEST_PROTOCOL)


def dump_variable(file_name, variable):
    with open(f'{file_name}.pickle', 'wb') as handle:
        pickle.dump(variable, handle, protocol=pickle.HIGHEST_PROTOCOL)


def load_variable(file_name):
    with open(f'{file_name}.pickle', 'rb') as handle:
        variable = pickle.load(handle)
    return variable


def get_frequencies():
    """
    Get frequencies used for collecting EIS data using pickle file.
    :return:
    """
    path = os.path.join(PROJECT_PATH(), 'Pickle data/frequencies.pickle')
    with open(path, 'rb') as input_file:
        frequencies = pickle.load(input_file)
    return frequencies.reshape((-1, 1))


def get_frequencies0(l=4, r=4, _range=400):
    """
    np.logspace customization.
    :param l:
    :param r:
    :param _range:
    :return:
    """
    return np.logspace(-l, r, _range).reshape((1, -1))


def extract_line(dataframe, index=0):
    """
    :param dataframe: Processed data
    :param index: chosen index
    :return: return desirable line from dataframe.
    """
    return np.array(dataframe.iloc[index]['Zmeas']).reshape((-1, 1))


def divide_signal(signal):
    """
    Divide GDRT Solution X
    :param signal:
    :return: R, L, C (Parasites), RC & RL
    """
    constant = 3
    length = (len(signal) - 3) // 2
    R, L, C = signal[0, 0], signal[1, 0], 1 / signal[2, 0]
    RC, RL = signal[constant: length + constant, :], signal[length + constant:, :]
    return R, L, C, RC, RL


def extract_peaks(signal, freq0):
    # TODO Enhance peaks extraction using differential techniques
    """

    :param signal:
    :param freq0:
    :return:
    """
    peaks = []
    for index in range(len(signal) - 2):
        if signal[index - 1] < signal[index] and signal[index + 1] < signal[index]:
            peaks.append({'mean': freq0[index], 'amplitude': signal[index]})
    return peaks


def _peaks(peaks):
    return np.array([peak['amplitude'] for peak in peaks]).reshape((len(peaks), 1)), np.array([peak['mean'] for peak in peaks]).reshape((len(peaks), 1))


def test():
    signal = np.array([i for i in range(104)])
    length = (len(signal) - 3) // 2
    print(length)
    l1 = list(range(1, length - 1))
    print(l1[0], l1[-1])
    l1 = list(range(1 + length, 2 * length - 1))
    print(l1[0], l1[-1])


if __name__ == '__main__':
    test()
