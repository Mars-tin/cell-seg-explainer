import os
import random
import numpy as np
from sklearn.metrics import accuracy_score

import torch

from model.mlp import MLP
from model.gcn import GCN
from model.gat import GAT
from dataset import generate_dataset, load_dataset


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def get_model(model_name, data, drop):
    if model_name == "gcn":
        model = GCN(num_features=data.num_features, num_classes=data.num_classes,
                    hidden_size=32, dropout=drop)
    elif model_name == "gat":
        model = GAT(num_features=data.num_features, num_classes=data.num_classes,
                    hidden_size=8, num_heads=4, dropout=drop)
    else:
        model = MLP(num_features=data.num_features, num_classes=data.num_classes,
                    hidden_size=32, dropout=drop)
    return model


def save_model(model, seed, epoch):
    checkpoint_dir = 'checkpoints'
    state = {
        'epoch': epoch,
        'state_dict': model.state_dict(),
    }
    filename = os.path.join(
        checkpoint_dir,
        'model_seed={}.tar'.format(seed)
    )
    torch.save(state, filename)


def load_model(model_name, data, seed, filename, cuda=False, pretrain=False):
    set_seed(seed)
    drop = 0
    if data == 'real':
        data = load_dataset()
    else:
        n_cell = 300
        n_edge = 3000
        data = generate_dataset(7, n_cell, n_edge, seed)
    model = get_model(model_name, data, drop)

    filename = os.path.join('checkpoints/', filename)
    if cuda:
        checkpoint = torch.load(filename)
    else:
        # Load GPU model on CPU
        checkpoint = torch.load(filename, map_location=lambda storage, loc: storage)
    epoch = checkpoint['epoch']
    model.load_state_dict(checkpoint['state_dict'], strict=not pretrain)
    return epoch, data, model


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