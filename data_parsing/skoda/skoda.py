import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt
from difflib import SequenceMatcher


LABEL_CODES = {
    32: 0,
    48: 1,
    49: 2,
    50: 3,
    51: 4,
    52: 5,
    53: 6,
    54: 7,
    55: 8,
    56: 9,
    57: 10
}


def split_signal(data, labels):
    split_idx = np.concatenate([np.where(np.diff(labels) != 0)[0] + 1])
    return (
        np.split(data, split_idx),
        [x[0] for x in np.split(labels, split_idx)]
    )


def main():
    left_data = loadmat('data/raw/skoda/left_classall_clean.mat')['left_classall_clean']
    right_data = loadmat('data/raw/skoda/right_classall_clean.mat')['right_classall_clean']

    col_indices = sum([[7 * i + 1, 7 * i + 2, 7 * i + 3] for i in range(10)], [])
    left_labels = np.vectorize(LABEL_CODES.get)(left_data[:, 0])
    right_labels = np.vectorize(LABEL_CODES.get)(right_data[:, 0])
    left_data = left_data[:, col_indices]
    right_data = right_data[:, col_indices]

    left_data, left_labels = split_signal(left_data, left_labels)
    right_data, right_labels = split_signal(right_data, right_labels)

    np.save('data/parsed/skoda/left',
            {'time_series': left_data, 'labels': left_labels}, allow_pickle=True)

    np.save('data/parsed/skoda/right',
            {'time_series': right_data, 'labels': right_labels}, allow_pickle=True)


if __name__ == '__main__':
    main()
