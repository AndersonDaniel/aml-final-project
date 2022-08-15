import numpy as np
import torch
from dataloaders.dataloader import SensorFusionDataset
from torch.utils.data import DataLoader
from models.attnsense import AttnSense
import json
from tqdm import tqdm
from sklearn.metrics import f1_score, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from collections import defaultdict


N_EPOCHS = 2000
# N_EPOCHS = 200
WINDOW_SIZE = 10
MINI_WINDOW_SIZE = 20

TRAIN_SUBJECTS = [101, 102, 105, 108, 109]
VAL_SUBJECTS = [106]
# SUBJECTS = [105]

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    with open('data/metadata.json', 'r') as f:
        metadata = json.load(f)

    train_datasets = [
        SensorFusionDataset(f'data/preprocessed/pamap2/{i}.npy', window_size=WINDOW_SIZE,
                            mini_window_size=MINI_WINDOW_SIZE)
        for i in TRAIN_SUBJECTS
    ]
    val_datasets = [
        SensorFusionDataset(f'data/preprocessed/pamap2/{i}.npy', window_size=WINDOW_SIZE,
                            mini_window_size=MINI_WINDOW_SIZE)
        for i in VAL_SUBJECTS
    ]
    train_ds = torch.utils.data.ConcatDataset(train_datasets)
    val_ds = torch.utils.data.ConcatDataset(val_datasets)
    # val_ds = torch.utils.data.Subset(val_ds, indices=[i for i in range(len(val_ds))
    #                                                   if not val_ds[i]['augmented']])
    
    train_dl = DataLoader(train_ds, shuffle=True, batch_size=64)
    val_dl = DataLoader(val_ds, shuffle=True, batch_size=64)
    model = AttnSense(metadata['pamap2']['sensor_groups'], train_datasets[0].labels.shape[1], train_datasets[0].n_freq,
                      device,
                      window_size=WINDOW_SIZE, mini_window_size=MINI_WINDOW_SIZE,
                      conv_dim=128).to(device)

    loss_f = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-6)
    scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=1e-6, max_lr=2e-4, cycle_momentum=False,
                                                  step_size_up=1000)
    
    epochs = range(1, N_EPOCHS + 1)
    epoch_losses = []
    evaluation_epochs = []
    train_epoch_f1_scores = []
    train_epoch_accuracies = []
    val_epoch_f1_scores = []
    val_epoch_accuracies = []

    for epoch in epochs:
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

        epoch_losses.append(np.mean(losses))

        if epoch % 20 == 0:
            evaluation_epochs.append(epoch)
            # train_acc, train_f1 = evaluate(model, train_dl, device)
            train_acc, train_f1 = evaluate_tta(model, train_dl, device)
            train_epoch_f1_scores.append(train_f1)
            train_epoch_accuracies.append(train_acc)
            # val_acc, val_f1 = evaluate(model, val_dl, device)
            val_acc, val_f1 = evaluate_tta(model, val_dl, device)
            val_epoch_f1_scores.append(val_f1)
            val_epoch_accuracies.append(val_acc)

    plt.plot(epochs, epoch_losses)
    plt.title('Train loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.savefig('results/loss.png')
    plt.close()

    plt.plot(evaluation_epochs, train_epoch_accuracies, label='train', marker='x')
    plt.plot(evaluation_epochs, val_epoch_accuracies, label='validation', marker='x')
    plt.title('Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig('results/accuracy.png')
    plt.close()

    plt.plot(evaluation_epochs, train_epoch_f1_scores, label='train', marker='x')
    plt.plot(evaluation_epochs, val_epoch_f1_scores, label='validation', marker='x')
    plt.title('F1 score')
    plt.xlabel('Epoch')
    plt.ylabel('F1')
    plt.legend()
    plt.savefig('results/f1.png')
    plt.close()

    # _, _, gt, pred = evaluate(model, val_dl, device, return_results=True)
    _, _, gt, pred = evaluate_tta(model, val_dl, device, return_results=True)
    make_confusion_matrix(gt, pred, metadata['pamap2']['label_names'])


def evaluate(model, dl, device, return_results=False):
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

    ret_val = ((gt == pred).mean(), f1_score(gt, pred, average='weighted'))
    if return_results:
        ret_val = (*ret_val, gt, pred)

    return ret_val


def evaluate_tta(model, dl, device, return_results=False):
    pbar = tqdm(iter(dl), desc=f'Evaluating')
    gt = {}
    pred = defaultdict(lambda: [])
    for batch in pbar:
        with torch.no_grad():
            curr_pred = model([x.to(device) for x in batch['spectrograms']]).detach().cpu().numpy()
            curr_gt = batch['label'].detach().cpu().numpy().argmax(axis=1)
            curr_ids = batch['id']

            for sample_id, sample_gt, sample_pred in zip(curr_ids, curr_gt, curr_pred):
                gt[sample_id] = sample_gt
                pred[sample_id].append(sample_pred)

    pred = {
        k: np.stack(v).mean(axis=0).argmax() for k, v in pred.items()
    }

    keys = gt.keys()
    gt = np.array([gt[k] for k in keys])
    pred = np.array([pred[k] for k in keys])

    print(f'accuracy: {(gt == pred).mean()}, f1: {f1_score(gt, pred, average="weighted")}')

    ret_val = ((gt == pred).mean(), f1_score(gt, pred, average='weighted'))
    if return_results:
        ret_val = (*ret_val, gt, pred)

    return ret_val


def make_confusion_matrix(gt, pred, label_names):
    unique_labels = np.unique(np.concatenate([gt, pred]))
    names = [label_names[str(l)] for l in unique_labels]

    fig, ax = plt.subplots(1, 1, figsize=(14, 14))

    ConfusionMatrixDisplay.from_predictions(gt, pred, display_labels=names,
                                            colorbar=False, cmap='Blues', normalize='true',
                                            ax=ax)

    plt.xticks(rotation=80)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Validation confusion matrix')
    plt.savefig('results/confusion_matrix.png')
    plt.close()


if __name__ == '__main__':
    main()
