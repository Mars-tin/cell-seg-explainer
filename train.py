from torch.optim import Adam
import torch.nn.functional as F

from utils import *
from dataset import generate_dataset, load_dataset


def train(model_name='gcn', data='synthetic', epochs=1000, seed=0, verbose=True):
    # Parameters
    log_interval = 10
    patience = 50
    lr = 1e-3
    err = 1e-4
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
    model = get_model(model_name, data, drop)

    # Optimizer
    optimizer = Adam(model.parameters(), lr=lr)

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
            if valid_loss < best_loss - err:
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
    save_model(model, seed, epoch)
    return epoch, data, best_model


if __name__ == "__main__":
    for seed in range(1):
        train(model_name='gcn', data='real', seed=seed)
        # train(model_name='gcn', data='synthetic', seed=seed)
