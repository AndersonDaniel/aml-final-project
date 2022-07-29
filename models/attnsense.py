import torch
import numpy as np


class SensorConv(torch.nn.Module):
    def __init__(self, group, conv_dim):
        super().__init__()
        self.conv = torch.nn.Sequential(
            torch.nn.Conv2d(len(group), 32, kernel_size=(3, 1)),
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(32),
            torch.nn.Conv2d(32, 32, kernel_size=(3, 1)),
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(32),
            torch.nn.MaxPool2d((2, 1)),
            torch.nn.Conv2d(32, conv_dim, kernel_size=(3, 1)),
            torch.nn.ReLU(),
            torch.nn.Flatten(start_dim=1, end_dim=2)
        )

    def forward(self, x):
        return self.conv(x)


class AttnSense(torch.nn.Module):
    def __init__(self, sensor_groups, n_classes, n_freq, conv_dim=64):
        super().__init__()

        self.sensor_convolutions = [
            SensorConv(group, conv_dim)
            for group in sensor_groups
        ]

        conv_embedding_dim = conv_dim * ((n_freq - 4) // 2 - 2)

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

        conv_res = torch.permute(conv_res, (0, 3, 1, 2))

        fusion_weights = self.sensor_fusion_attention(conv_res)[..., 0]

        fusion_weights = torch.softmax(fusion_weights, dim=-1)
        fusion_weights = fusion_weights.reshape((*fusion_weights.shape, 1)).expand(conv_res.shape)

        timestep_embeddings = (conv_res * fusion_weights).sum(dim=2)

        timestep_res = self.gru(timestep_embeddings)[0]

        timestep_weights = torch.softmax(self.time_attention(timestep_res)[..., 0], dim=-1)
        timestep_weights = timestep_weights.reshape((*timestep_weights.shape, 1)).expand(timestep_res.shape)

        embedding = (timestep_res * timestep_weights).sum(dim=1)

        return self.classification_head(embedding)
