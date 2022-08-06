import numpy as np
import torch
from dataloaders.dataloader import SensorFusionDataset
from torch.utils.data import DataLoader
from models.attnsense import AttnSense
import json
from tqdm import tqdm
from sklearn.metrics import f1_score


N_EPOCHS = 10000
WINDOW_SIZE = 10
MINI_WINDOW_SIZE = 20

# SUBJECTS = [101, 102, 103, 104, 105, 106, 107, 108, 109]
# SUBJECTS = [101, 102, 105, 106, 108, 109]
SUBJECTS = [105]

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    with open('data/metadata.json', 'r') as f:
        metadata = json.load(f)

    # ds = SensorFusionDataset('data/preprocessed/pamap2/105.npy', window_size=WINDOW_SIZE,
    #                          mini_window_size=MINI_WINDOW_SIZE)
    datasets = [
        SensorFusionDataset(f'data/preprocessed/pamap2/{i}.npy', window_size=WINDOW_SIZE,
                            mini_window_size=MINI_WINDOW_SIZE)
        for i in SUBJECTS
    ]
    ds = torch.utils.data.ConcatDataset(datasets)
    n_train = int(.75 * len(ds))
    n_val = len(ds) - n_train
    train_ds, val_ds = torch.utils.data.random_split(ds, [n_train, n_val])
    
    # ds = SensorFusionDataset('data/preprocessed/skoda/left.npy')
    train_dl = DataLoader(train_ds, shuffle=True, batch_size=64)
    val_dl = DataLoader(val_ds, shuffle=True, batch_size=64)
    model = AttnSense(metadata['pamap2']['sensor_groups'], datasets[0].labels.shape[1], datasets[0].n_freq,
                      device,
                      window_size=WINDOW_SIZE, mini_window_size=MINI_WINDOW_SIZE,
                      conv_dim=64).to(device)

    # model = AttnSense(metadata['skoda']['sensor_groups'], ds.labels.shape[1])
    loss_f = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)
    # optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=1e-4, max_lr=2e-3, cycle_momentum=False,
                                                  step_size_up=1000)
    # optimizer = torch.optim.Adam(model.parameters(), lr=2e-3)
    # optimizer = torch.optim.RMSprop(model.parameters(), lr=5e-4)

    for epoch in range(1, N_EPOCHS + 1):
        pbar = tqdm(iter(train_dl), desc=f'Epoch {epoch}/{N_EPOCHS}')
        losses = []
        for batch in pbar:
            pred = model([x.to(device) for x in batch['spectrograms']])

            optimizer.zero_grad()
            loss = loss_f(pred, batch['label'].to(device))
            loss.backward()
            optimizer.step()
            scheduler.step()
            
            losses.append(loss.item())
            pbar.set_postfix({'loss': np.mean(losses)})

        if epoch % 20 == 0:
            evaluate(model, val_dl, device)


def evaluate(model, dl, device):
    pbar = tqdm(iter(dl), desc=f'Evaluating')
    gt = None
    pred = None
    for batch in pbar:
        with torch.no_grad():
            curr_pred = model([x.to(device) for x in batch['spectrograms']]).detach().cpu().numpy().argmax(axis=1)
            curr_gt = batch['label'].detach().cpu().numpy().argmax(axis=1)

            if gt is None:
                gt = curr_gt
                pred = curr_pred
            else:
                gt = np.concatenate([gt, curr_gt])
                pred = np.concatenate([pred, curr_pred])

        pbar.set_postfix({'accuracy': (gt == pred).mean(), 'f1': f1_score(gt, pred, average='weighted')})



if __name__ == '__main__':
    main()
