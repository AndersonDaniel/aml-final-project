import numpy as np
import matplotlib.pyplot as plt


def main():
    x1 = np.load('data/parsed/har_hetero/d.npy', allow_pickle=True).item()
    print(x1['time_series'][0].shape)

    x2 = np.load('data/parsed/skoda/left.npy', allow_pickle=True).item()
    print(x2['time_series'][0].shape)

    x3 = np.load('data/parsed/pamap2/103.npy', allow_pickle=True).item()
    print(x3['time_series'][0].shape)


if __name__ == '__main__':
    main()
