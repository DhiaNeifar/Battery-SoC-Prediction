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


def PICKLED_DATA_PATH():
    return os.path.join(PROJECT_PATH(), 'Pickle Data')


def CSV_DATA(filename):
    return os.path.join(PROJECT_PATH(), 'Data', f'{filename}.csv')


def unprocessed_pickled_data():
    pickle_list = os.listdir(PICKLED_DATA_PATH())
    frequencies = 'frequencies'
    new_list = []
    for pickle_file in pickle_list:
        file = pickle_file[:-7]
        if file != frequencies:
            new_list.append(file)
    return new_list


def dump_frequencies():
    f = np.array([0.01000, 0.03667, 0.05000, 0.05667, 0.06333, 0.07000, 0.08000, 0.09000, 0.10000, 0.11333, 0.12667, 0.14000, 0.16000, 0.17667, 0.20000, 0.22333, 0.25000, 0.28333, 0.31667, 0.35333, 0.39667, 0.44667, 0.50000, 0.56333, 0.63000, 0.70667, 0.79333, 0.89000, 1.00000, 1.12333, 1.26000, 1.41333, 1.58333, 1.77667, 1.99667, 2.24000, 2.51333, 2.82000, 3.16333, 3.54667, 3.98000, 4.46667, 5.01333, 5.62333, 6.31000, 7.08000, 7.94333, 8.91333, 10.00000, 11.22000, 12.59000, 14.12667, 15.85000, 17.78333, 19.95333, 22.38667, 25.12000, 28.18333, 31.62333, 35.48000, 39.81000, 44.66667, 50.12000, 56.23333, 63.09667, 70.79333, 79.43333, 89.12333, 100.00000, 112.20333, 125.89333, 141.25333, 158.49000, 177.82667, 199.52667, 223.87333, 251.18667, 281.83667, 316.22667, 354.81333, 398.10667, 446.68333, 501.18667, 562.34000, 630.95667, 707.94333, 794.32667, 891.25000, 1000.00000])
    print(f.shape)
    with open(os.path.join(PROJECT_PATH(), 'Pickle Data', 'frequencies.pickle'), 'wb') as handle:
        pickle.dump(f, handle, protocol=pickle.HIGHEST_PROTOCOL)


def dump_variable(file_name, variable):
    with open(os.path.join(PROJECT_PATH(), 'Pickle Data', f'{file_name}.pickle'), 'wb') as handle:
        pickle.dump(variable, handle, protocol=pickle.HIGHEST_PROTOCOL)


def load_variable(file_name):
    with open(os.path.join(PROJECT_PATH(), 'Pickle Data', f'{file_name}.pickle'), 'rb') as handle:
        variable = pickle.load(handle)
    return variable


def get_frequencies():
    """
    Get frequencies used for collecting EIS data using pickle file.
    :return:
    """
    with open(os.path.join(PROJECT_PATH(), 'Pickle Data', 'frequencies.pickle'), 'rb') as input_file:
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
    :param index: Chosen index
    :return: return desirable line (Z IMPEDANCE) from dataframe.
    """
    print(dataframe.iloc[index][['CellID', 'SoC', 'Cycle']])
    return np.array(dataframe.iloc[index]['Zmeas']).reshape((-1, 1))


def divide_signal(signal, parasites=True):
    """
    Divide GDRT Solution X
    :param signal:
    :param parasites: If parasites are into equation
    :return: R, L, C (Parasites), RC & RL
    """
    constant = 1
    if parasites:
        constant = 3
    length = (len(signal) - constant) // 2
    L, C = signal[1, 0], 1 / signal[2, 0]
    RC, RL = signal[constant: length + constant, :], signal[length + constant:, :]
    return L, C, RC, RL


def extract_peaks(signal, f_GDRT):
    # TODO : Enhance peaks extraction using differential techniques
    """
    Find peaks in a signal.
    :param signal:
    :param f_GDRT:
    :return: Dictionary with means e.i frequencies and their amplitudes.
    """
    signal, f_GDRT = np.squeeze(signal), np.squeeze(f_GDRT)
    peaks = []
    for index in range(len(signal) - 2):
        if signal[index - 1] < signal[index] and signal[index + 1] < signal[index]:
            peaks.append({'mean': f_GDRT[index], 'amplitude': signal[index]})
    return peaks


def amplitudes_means_std_from_peaks(peaks):
    if peaks:
        if len(peaks[0]) == 2:
            return np.array([peak['amplitude'] for peak in peaks]).reshape((len(peaks), 1)), \
                np.array([peak['mean'] for peak in peaks]).reshape((len(peaks), 1)), \
                np.zeros((len(peaks), 1))

        if len(peaks[0]) == 3:
            return np.array([peak['amplitude'] for peak in peaks]).reshape((len(peaks), 1)), \
                np.array([peak['mean'] for peak in peaks]).reshape((len(peaks), 1)), \
                np.array([peak['std'] for peak in peaks]).reshape((len(peaks), 1))

    null = np.zeros((1, 1))
    return null, null, null


def amplitudes_means_standard_deviations_from_df(line, peaks=7):
    amplitudes = np.array(line[[f'RC_amplitude_{i}' for i in range(1, peaks + 1)]]).reshape((-1, 1))
    means = np.array(line[[f'RC_mean_{i}' for i in range(1, peaks + 1)]]).reshape((-1, 1))
    standard_deviations = np.array(line[[f'RC_standard_deviation_{i}' for i in range(1, peaks + 1)]]).reshape((-1, 1))
    return amplitudes[~np.all(amplitudes == 0, axis=1)].astype('float64'), means[~np.all(means == 0, axis=1)].astype('float64'), standard_deviations[~np.all(standard_deviations == 0, axis=1)].astype('float64')


def amp_means_std_to_peaks(amplitudes, means, standard_deviations):
    return [{'mean': means[i, 0], 'amplitude':amplitudes[i, 0], 'std': standard_deviations[i, 0]} for i in range(amplitudes.shape[0])]


def find_std(means):
    new_stds = np.zeros(means.shape)
    rows = means.shape[0]
    if rows == 1:
        return 0.03 * np.ones((1, 1))
    x = (means[1, 0] + means[0, 0]) / 2
    new_stds[0, 0] = x / 3
    i = 2
    while i < rows:
        x1 = (means[i, 0] + means[i - 1, 0]) / 2
        if x == 0:
            new_stds[i - 1, 0] = x1 / 3
        elif x1 == 0:
            new_stds[i - 1, 0] = x / 3
        else:
            new_stds[i - 1, 0] = (x + x1) / 6
        x = x1
        i += 1
    new_stds[i - 1, 0] = x / 3
    return new_stds


def normal_distribution(f_GDRT, amplitudes, means, standard_deviations):
    return (amplitudes * np.exp(-0.5 * ((f_GDRT - means) / standard_deviations) ** 2)).T


def extract_distributions(dataframe, index):
    line = dataframe.iloc[index]
    peaks = int(line['RC_peaks'])
    amplitudes, means, standard_divs = np.ones((peaks, 1)), np.ones((peaks, 1)), np.ones((peaks, 1))
    for index, peak in enumerate(range(peaks), 1):
        amplitudes[peak, 0] = line[f'RC_amplitude_{index}']
        means[peak, 0] = line[f'RC_mean_{index}']
        standard_divs[peak, 0] = line[f'RC_standard_deviation_{index}']
    return amplitudes, means, standard_divs


def find_max(variable):
    maximum = variable[0].shape[0]
    for i in range(1, len(variable)):
        x = variable[i].shape[0]
        maximum = max(x, maximum)
    return maximum
