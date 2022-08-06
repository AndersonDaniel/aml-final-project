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

    # preprocess('data/parsed/pamap2/101.npy', metadata['pamap2'])


def preprocess(path, metadata):
    x = np.load(path, allow_pickle=True).item()
    spectrograms = []
    labels = []
    augmented = []
    for time_series, label in zip(x['time_series'], x['labels']):
        # spectrograms.append(get_spectrograms(time_series, metadata['freq_hz'], metadata['sensor_groups']))
        spectrograms.append(get_spectrograms(time_series, metadata['freq_hz'], metadata['sensor_groups'],
                                             window_sec=.2))
        labels.append(label)
        augmented.append(False)
        for i in range(3):
            augmented_time_series = augment_time_series(time_series)
            spectrograms.append(get_spectrograms(augmented_time_series, metadata['freq_hz'], metadata['sensor_groups'],
                                                 window_sec=.2))
            labels.append(label)
            augmented.append(True)

    np.save(path.replace('/parsed/', '/preprocessed/'),
            {'spectrograms': spectrograms, 'labels': labels, 'augmented': augmented})


def augment_time_series(x):
    return x + np.random.normal(loc=0, scale=x.std(axis=0) / 5, size=x.shape)


# def get_spectrograms(signals, freq_hz, signal_groups, window_sec=.25):
#     spectrograms = [
#         np.abs(stft(signals[..., i], nperseg=round(window_sec * freq_hz), fs=freq_hz)[2])
#         for i in range(signals.shape[1])
#     ]
#
#     return [
#         np.stack([spectrograms[idx] for idx in group], axis=-1)
#         for group in signal_groups
#     ]


def get_spectrograms(signals, freq_hz, signal_groups, window_sec=.25):
    signals = [np.linalg.norm(signals[:, group], axis=1) for group in signal_groups]
    return [
        np.abs(stft(signals[i], nperseg=round(window_sec * freq_hz), fs=freq_hz, nfft=32)[2])[..., np.newaxis]
        for i in range(len(signals))
    ]


if __name__ == '__main__':
    main()
