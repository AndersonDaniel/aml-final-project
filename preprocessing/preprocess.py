import numpy as np
import json
from scipy.signal import stft
from tqdm import tqdm


def main():
    with open('data/metadata.json', 'r') as f:
        metadata = json.load(f)

    for p in tqdm('abcdefghi'):
        preprocess(f'data/parsed/har_hetero/{p}.npy', metadata['har_hetero'])

    preprocess('data/parsed/skoda/left.npy', metadata['skoda'])
    preprocess('data/parsed/skoda/right.npy', metadata['skoda'])

    for s in tqdm(range(101, 110)):
        preprocess(f'data/parsed/pamap2/{s}.npy', metadata['pamap2'])


def preprocess(path, metadata):
    x = np.load(path, allow_pickle=True).item()
    spectrograms = []
    for time_series in x['time_series']:
        spectrograms.append(get_spectrograms(time_series, metadata['freq_hz'], metadata['sensor_groups']))

    np.save(path.replace('/parsed/', '/preprocessed/'),
            {'spectrograms': spectrograms, 'labels': x['labels']})


def get_spectrograms(signals, freq_hz, signal_groups, window_sec=.25):
    spectrograms = [
        np.abs(stft(signals[..., i], nperseg=round(window_sec * freq_hz), fs=freq_hz, nfft=128)[2])
        for i in range(signals.shape[1])
    ]

    return [
        np.stack([spectrograms[idx] for idx in group], axis=-1)
        for group in signal_groups
    ]


if __name__ == '__main__':
    main()
