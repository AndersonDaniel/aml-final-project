import numpy as np
import torch
from dataloaders.dataloader import SensorFusionDataset
from torch.utils.data import DataLoader
from models.attnsense import AttnSense
import json
from tqdm import tqdm


N_EPOCHS = 200


def main():
    with open('data/metadata.json', 'r') as f:
        metadata = json.load(f)

    ds = SensorFusionDataset('data/preprocessed/pamap2/105.npy')
    # ds = SensorFusionDataset('data/preprocessed/skoda/left.npy')
    dl = DataLoader(ds, shuffle=True, batch_size=64)
    model = AttnSense(metadata['pamap2']['sensor_groups'], ds.labels.shape[1], ds.n_freq)
    # model = AttnSense(metadata['skoda']['sensor_groups'], ds.labels.shape[1])
    loss_f = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    for epoch in range(1, N_EPOCHS + 1):
        pbar = tqdm(iter(dl), desc=f'Epoch {epoch}/{N_EPOCHS}')
        losses = []
        for batch in pbar:
            pred = model(batch['spectrograms'])

            optimizer.zero_grad()
            loss = loss_f(pred, batch['label'])
            loss.backward()
            optimizer.step()

            losses.append(loss.item())
            pbar.set_postfix({'loss': np.mean(losses)})


if __name__ == '__main__':
    main()