import torch
import random
import numpy as np


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def log_training(epoch, stats):
    valid_loss, train_loss, _ = stats[-1]
    print('Epoch {}'.format(epoch))
    print('\tValidation Loss: {}'.format(valid_loss))
    print('\tTrain Loss: {}'.format(train_loss))


def train(data, model, loss_fn, optimizer):
    model.train()
    optimizer.zero_grad()
    output = model(data)
    y_pred = output[data.train_mask]
    y_true = data.y[data.train_mask]
    loss = loss_fn(y_pred, y_true)
    loss.backward()
    optimizer.step()


def get_loss(data, model, criterion):
    model.eval()
    with torch.no_grad():
        output = model(data)
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
