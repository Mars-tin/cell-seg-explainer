import argparse
from torch.optim import Adam

from model.mlp import MLP
from model.gcn import GCN
from model.gat import GAT
from generate import generate_dataset
from utils import *


def main(seed):
    # Parameters
    verbose = True
    log_interval = 10

    model_name = "gcn"
    epochs = 1000
    patience = 50
    n_cell = 300
    n_edge = 5000
    drop = 0

    # Dataset
    set_seed(seed)
    data = generate_dataset(n_cell, n_edge, seed)

    # Loss functions
    criterion = torch.nn.MSELoss()
    loss_fn = torch.nn.MSELoss()

    # Model
    if model_name == "gcn":
        model = GCN(
            num_features=7,
            hidden_size=32,
            dropout=drop
        )
    elif model_name == "gat":
        model = GAT(
            num_features=7,
            hidden_size=4,
            num_heads=8,
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
    wait = patience
    idx = -1
    stats = []

    # Training
    for epoch in range(epochs):
        # Early stop
        if wait < 0:
            break

        # Train model
        train(data, model, loss_fn, optimizer)

        # Evaluate model
        if (epoch + 1) % log_interval == 0:
            evaluate(data, model, criterion, epoch+1, stats, verbose)
            valid_loss = stats[-1][0]
            if valid_loss < best_loss:
                wait = patience
                best_loss = valid_loss
                idx = epoch
            wait -= 1

    epoch = idx
    idx = min(int((idx+1)/log_interval), len(stats)-1)
    print("The loss on test dataset is:", stats[idx][2], "obtained in epoch", epoch)


if __name__ == "__main__":
    # parser = argparse.ArgumentParser()
    # args, _ = parser.parse_known_args()
    for seed in range(10):
        main(seed)