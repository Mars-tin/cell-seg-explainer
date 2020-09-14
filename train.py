import random
import numpy as np
from sklearn.metrics import accuracy_score

import torch
from torch.optim import Adam
import torch.nn.functional as F

from model.mlp import MLP
from model.gcn import GCN
from model.gat import GAT
from dataset import generate_dataset, load_dataset


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def log_training(epoch, stats):
    valid_loss, train_loss, _, valid_acc, _ = stats[-1]
    print('Epoch {}'.format(epoch))
    print('\tTrain Loss: {}'.format(train_loss))
    print('\tValidation Loss: {}'.format(valid_loss))
    print('\tValidation Accuracy: {}'.format(valid_acc))


def train_epoch(data, model, loss_fn, optimizer):
    model.train()
    optimizer.zero_grad()
    output = model(data.X, data.edge_index)
    y_pred = output[data.train_mask]
    y_true = data.y.squeeze()[data.train_mask]
    loss = loss_fn(y_pred, y_true)
    loss.backward()
    optimizer.step()


def get_stats(data, model, criterion):
    with torch.no_grad():
        logits = model(data.X, data.edge_index)
        y_pred = logits.max(1)[1]
        y_true = data.y.squeeze()

        loss_train = criterion(
            logits[data.train_mask],
            y_true[data.train_mask]).item()
        loss_valid = criterion(
            logits[data.val_mask],
            y_true[data.val_mask]).item()
        loss_test = criterion(
            logits[data.test_mask],
            y_true[data.test_mask]).item()

        acc_valid = accuracy_score(y_true[data.val_mask], y_pred[data.val_mask])
        acc_test = accuracy_score(y_true[data.test_mask], y_pred[data.test_mask])

    return loss_train, loss_valid, loss_test, acc_valid, acc_test


def evaluate(data, model, criterion, epoch, stats, verbose=False):
    model.eval()
    train_loss, val_loss, test_loss, valid_acc, test_acc = get_stats(data, model, criterion)
    stats.append([val_loss, train_loss, test_loss, valid_acc, test_acc])
    if verbose:
        log_training(epoch, stats)


def train(model_name='gcn', data='synthetic', epochs=1000, seed=0, verbose=True):
    # Parameters
    log_interval = 10
    patience = 50
    drop = 0

    # Dataset
    set_seed(seed)
    if data == 'real':
        data = load_dataset()
    else:
        n_cell = 300
        n_edge = 3000
        data = generate_dataset(7, n_cell, n_edge, seed)

    # Loss functions
    criterion = F.cross_entropy
    loss_fn = F.cross_entropy

    # Model
    if model_name == "gcn":
        model = GCN(
            num_features=data.num_features,
            num_classes=data.num_classes,
            hidden_size=32,
            dropout=drop
        )
    elif model_name == "gat":
        model = GAT(
            num_features=data.num_features,
            num_classes=data.num_classes,
            hidden_size=8,
            num_heads=4,
            dropout=drop
        )
    else:
        model = MLP(
            num_features=data.num_features,
            num_classes=data.num_classes,
            hidden_size=32,
            dropout=drop
        )

    # Optimizer
    optimizer = Adam(model.parameters(), lr=1e-3)

    # Early stop setup
    best_loss = float('inf')
    best_model = None
    wait = patience
    idx = -1
    stats = []

    # Training
    for epoch in range(epochs):
        # Early stop
        if wait < 0:
            break

        # Train model
        train_epoch(data, model, loss_fn, optimizer)

        # Evaluate model
        if (epoch + 1) % log_interval == 0:
            evaluate(data, model, criterion, epoch+1, stats, verbose)
            valid_loss = stats[-1][0]
            if valid_loss < best_loss:
                wait = patience
                best_loss = valid_loss
                best_model = model
                idx = epoch
            wait -= 1

    epoch = idx
    idx = min(int((idx+1)/log_interval), len(stats)-1)
    print("The loss on test dataset is:", stats[idx][2],
          "| The accuracy on test dataset is:", stats[idx][-1],
          "| Obtained in epoch", epoch)
    return epoch, data, best_model


if __name__ == "__main__":
    for seed in range(1):
        train(model_name='gcn', data='real', seed=seed)
        # train(model_name='gcn', data='synthetic', seed=seed)
