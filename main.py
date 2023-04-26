from utils import get_frequencies, get_frequencies0
from Data.data import import_data, save_data


def main():
    data = import_data()
    f = get_frequencies()
    f_GDRT = get_frequencies0()
    save_data(data, f, f_GDRT)


if __name__ == '__main__':
    main()
