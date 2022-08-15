import numpy as np
import json
from scipy.signal import stft
from tqdm import tqdm
from transforms3d.axangles import axangle2mat
from uuid import uuid4


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
    augmentations = [augmentation_jitter, augmentation_scale, augmentation_jitter_scale]
    spectrograms = []
    labels = []
    augmented = []
    ids = []
    for time_series, label in zip(x['time_series'], x['labels']):
        curr_id = str(uuid4())
        spectrograms.append(get_spectrograms(time_series, metadata['freq_hz'], metadata['sensor_groups'],
                                             window_sec=.2))
        labels.append(label)
        augmented.append(False)
        ids.append(curr_id)
        for i in range(5):
            augmentation = np.random.choice(augmentations)
#         for augmentation in augmentations:
#             for i in range(3):
            augmented_time_series = augmentation(time_series)
            spectrograms.append(get_spectrograms(augmented_time_series, metadata['freq_hz'], metadata['sensor_groups'],
                                                 window_sec=.2))
            labels.append(label)
            augmented.append(True)
            ids.append(curr_id)

    np.save(path.replace('/parsed/', '/preprocessed/'),
            {'spectrograms': spectrograms, 'labels': labels, 'augmented': augmented, 'ids': ids})


# def augment_time_series(x):
#     return x + np.random.normal(loc=0, scale=x.std(axis=0) / 5, size=x.shape)

def augmentation_jitter(x):
    return x + np.random.normal(loc=0, scale=x.std(axis=0) / 5, size=x.shape)

def augmentation_scale(x, sigma=0.1):
    return x * np.random.normal(loc=1.0, scale=sigma, size=(1,x.shape[1]))

def augmentation_rotation(x):
    axis = np.random.uniform(low=-1, high=1, size=x.shape[1])
    angle = np.random.uniform(low=-np.pi, high=np.pi)
    return np.matmul(x, axangle2mat(axis,angle))

def augmentation_jitter_rotation(x):
    return augmentation_jitter(augmentation_rotation(x))

def augmentation_jitter_scale(x):
    return augmentation_jitter(augmentation_scale(x))

def augmentation_scale_rotation(x):
    return augmentation_scale(augmentation_rotation(x))

def get_spectrograms(signals, freq_hz, signal_groups, window_sec=.25):
    signals = [np.linalg.norm(signals[:, group], axis=1) for group in signal_groups]
    return [
        np.abs(stft(signals[i], nperseg=round(window_sec * freq_hz), fs=freq_hz, nfft=32)[2])[..., np.newaxis]
        for i in range(len(signals))
    ]


if __name__ == '__main__':
    main()
