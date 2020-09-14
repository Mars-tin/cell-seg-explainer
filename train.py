from torch.optim import Adam
import torch
import random
import numpy as np

from model.mlp import MLP
from model.gcn import GCN
from model.gat import GAT
from dataset import generate_dataset, load_dataset


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def log_training(epoch, stats):
    valid_loss, train_loss, _ = stats[-1]
    print('Epoch {}'.format(epoch))
    print('\tValidation Loss: {}'.format(valid_loss))
    print('\tTrain Loss: {}'.format(train_loss))


def train_epoch(data, model, loss_fn, optimizer):
    model.train()
    optimizer.zero_grad()
    output = model(data.X, data.edge_index)
    y_pred = output[data.train_mask]
    y_true = data.y[data.train_mask]
    loss = loss_fn(y_pred, y_true)
    loss.backward()
    optimizer.step()


def get_loss(data, model, criterion):
    model.eval()
    with torch.no_grad():
        output = model(data.X, data.edge_index)
        loss_train = criterion(
            output[data.train_mask],
            data.y[data.train_mask]).item()
        loss_valid = criterion(
            output[data.val_mask],
            data.y[data.val_mask]).item()
        loss_test = criterion(
            output[data.test_mask],
            data.y[data.test_mask]).item()
    return loss_train, loss_valid, loss_test


def evaluate(data, model, criterion, epoch, stats, verbose=False):
    train_loss, val_loss, test_loss = get_loss(data, model, criterion)
    stats.append([val_loss, train_loss, test_loss])
    if verbose:
        log_training(epoch, stats)


def train(model_name='gcn', data='synthetic', epochs=1000, seed=0):
    # Parameters
    verbose = False
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
        data = generate_dataset(n_cell, n_edge, seed)

    # Loss functions
    criterion = torch.nn.MSELoss()
    loss_fn = torch.nn.MSELoss()

    # Model
    if model_name == "gcn":
        model = GCN(
            num_features=data.num_features,
            hidden_size=32,
            dropout=drop
        )
    elif model_name == "gat":
        model = GAT(
            num_features=7,
            hidden_size=8,
            num_heads=4,
            dropout=drop
        )
    else:
        model = MLP(
            num_features=7,
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
    print("The loss on test dataset is:", stats[idx][2], "obtained in epoch", epoch)
    return epoch, data, best_model


if __name__ == "__main__":
    for seed in range(10):
        train(model_name='gcn', seed=seed)
