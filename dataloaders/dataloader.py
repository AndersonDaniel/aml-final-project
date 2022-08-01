from torch.utils.data import Dataset, DataLoader
import torch
import numpy as np


class SensorFusionDataset(Dataset):
    def __init__(self, data_path, window_size=10, mini_window_size=20, overlap=.25):
        data = np.load(data_path, allow_pickle=True).item()
        self.labels = torch.nn.functional.one_hot(torch.tensor(np.array(data['labels']))).to(torch.float32)
        self.spectrograms = data['spectrograms']
        self.window_size = window_size
        self.mini_window_size = mini_window_size
        self.full_window_size = window_size * mini_window_size
        self.overlap = int(overlap * window_size)
        self.lengths = [
            (self.spectrograms[i][0].shape[1] - self.full_window_size) // (self.full_window_size - self.overlap)
            for i in range(len(self.spectrograms))
        ]
        self.n_freq = self.spectrograms[0][0].shape[0]

    def __len__(self):
        return sum(self.lengths)

    def __getitem__(self, idx):
        cum_lengths = np.cumsum(self.lengths)
        session_idx = np.where(cum_lengths > idx)[0][0]
        sample_idx = idx
        if session_idx > 0:
            sample_idx -= cum_lengths[session_idx - 1]

        window_start = sample_idx * (self.full_window_size - self.overlap)
        full_window = [
            s[:, window_start:window_start + self.full_window_size].transpose((2, 0, 1)).astype(np.float32)
            for s in self.spectrograms[session_idx]
        ]

        # window = [w.reshape((*w.shape[:2], self.window_size, self.mini_window_size)) for w in full_window]

        return {
            'spectrograms': full_window,
            'label': self.labels[session_idx]
        }
