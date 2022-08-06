import torch
import numpy as np


class SensorConv(torch.nn.Module):
    def init_weights(self, m):
        if isinstance(m, torch.nn.Linear) or isinstance(m, torch.nn.Conv2d):
            torch.nn.init.normal_(m.weight, 0, 0.02)

    def __init__(self, group, n_freq, conv_dim, window_size):
        super().__init__()
        self.window_size = window_size
        self.conv = torch.nn.Sequential(
            torch.nn.Conv2d(1, 64, kernel_size=(n_freq, 3), padding=(0, 1)),
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(64),
            torch.nn.Conv2d(64, 64, kernel_size=(1, 3), padding=(0, 1)),
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(64),
            torch.nn.MaxPool2d((1, 2)),
            torch.nn.Conv2d(64, conv_dim, kernel_size=(1, 3), padding=(0, 1)),
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(conv_dim),
            torch.nn.Conv2d(conv_dim, conv_dim, kernel_size=(1, 3), padding=(0, 1)),
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(conv_dim),
            torch.nn.MaxPool2d((1, 2)),
            torch.nn.Conv2d(conv_dim, conv_dim, kernel_size=(1, 3), padding=(0, 1)),
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(conv_dim),
            torch.nn.Flatten(start_dim=1, end_dim=2)
        )

        self.conv.apply(self.init_weights)

    def forward(self, x):
        res = self.conv(x)
        res = torch.permute(torch.reshape(res, (*res.shape[:-1], self.window_size, -1)), (0, 2, 1, 3))
        res = torch.flatten(res, start_dim=2, end_dim=3)

        return res


class AttnSense(torch.nn.Module):
    def __init__(self, sensor_groups, n_classes, n_freq, device, conv_dim=32,
                 window_size=10, mini_window_size=20):
        super().__init__()
        self.window_size = window_size

        self.sensor_convolutions = [
            SensorConv(group, n_freq, conv_dim, window_size).to(device)
            for group in sensor_groups
        ]

        conv_embedding_dim = conv_dim * (mini_window_size // 4)

        self.sensor_fusion_attention = torch.nn.Sequential(
            torch.nn.Linear(conv_embedding_dim, 1),
            torch.nn.Tanh(),
        )

        self.gru = torch.nn.GRU(
            conv_embedding_dim, 64, 2, batch_first=True
        )

        self.time_attention = torch.nn.Sequential(
            torch.nn.Linear(64, 1),
            torch.nn.Tanh()
        )

        self.classification_head = torch.nn.Sequential(
            torch.nn.Linear(64, n_classes),
            torch.nn.Softmax(dim=-1)
        )

    def forward(self, X):
        conv_res = torch.stack([
            conv(x)
            for x, conv in zip(X, self.sensor_convolutions)
        ], dim=1)

        fusion_weights = self.sensor_fusion_attention(conv_res)[..., 0]

        fusion_weights = torch.softmax(fusion_weights, dim=-1)
        fusion_weights = fusion_weights.reshape((*fusion_weights.shape, 1)).expand(conv_res.shape)

        timestep_embeddings = (conv_res * fusion_weights).sum(dim=2)

        timestep_res = self.gru(timestep_embeddings)[0]

        timestep_weights = torch.softmax(self.time_attention(timestep_res)[..., 0], dim=-1)
        timestep_weights = timestep_weights.reshape((*timestep_weights.shape, 1)).expand(timestep_res.shape)

        embedding = (timestep_res * timestep_weights).sum(dim=1)

        res = self.classification_head(embedding)

        return res
