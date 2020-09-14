import os
import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection
from matplotlib.patches import Patch, Rectangle

import torch
from torch.utils.data import Dataset


class SyntheticDataset(Dataset):

    def __init__(self, num_classes, num_features, loc, marker, label, n_edge=None):
        super().__init__()
        self.num_classes = num_classes
        self.num_features = num_features

        self.X = torch.from_numpy(marker).float()
        self.y = torch.from_numpy(label).long()
        n = self.__len__()

        if n_edge is None:
            n_edge = 10 * n
        x = np.mean(loc[0:2], axis=0)
        y = np.mean(loc[2:4], axis=0)
        center = np.concatenate((x, y))
        logits = -np.linalg.norm(
            center.reshape(1, n, 2) - center.reshape(n, 1, 2), axis=2
        )
        threshold = np.sort(logits.reshape(-1))[-n_edge-n]
        adj = (logits >= threshold).astype(float) - np.eye(n)
        self.edge_index = torch.tensor(np.array(list(adj.nonzero())))

        n = self.__len__()
        m = n // 3
        self.train_mask = torch.zeros(n).to(dtype=torch.bool)
        self.train_mask[:m] = True
        self.val_mask = torch.zeros(n).to(dtype=torch.bool)
        self.val_mask[m:2*m] = True
        self.test_mask = torch.zeros(n).to(dtype=torch.bool)
        self.test_mask[-m:] = True

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

    def merge(self, dataset):
        new_edge_index = dataset.edge_index + self.__len__()
        self.edge_index = torch.cat((self.edge_index, new_edge_index), 1)
        self.X = torch.cat((self.X, dataset.X), 0)
        self.y = torch.cat((self.y, dataset.y), 0)
        self.train_mask = torch.cat((self.train_mask, dataset.train_mask), 0)
        self.val_mask = torch.cat((self.val_mask, dataset.val_mask), 0)
        self.test_mask = torch.cat((self.test_mask, dataset.test_mask), 0)


def generate_sample(num_features=7, size=300, seed=0, save=False):
    """
    Cell information:
    * Location: square box, x, y, r1, r2
    * Marker: a vector of size 7
    * Label: +1 and -1
    """
    # Parameters
    filename = 'cell_sample_size={}_seed={}.pkl'.format(size, seed)
    fig_size = 500
    marker_mu = 3
    marker_sigma = 1
    radius_mu = 10
    radius_sigma = 1
    n_edge = 5000

    # Generate features
    rs = np.random.RandomState(seed=seed)
    cen = rs.uniform(0, fig_size, size=(size, 2))
    rad = np.abs(rs.normal(radius_mu, radius_sigma, size=(size, 2)))
    loc = np.array([cen[:, 0]-rad[:, 0], cen[:, 0]+rad[:, 0], cen[:, 1]-rad[:, 1], cen[:, 1]+rad[:, 1]])
    marker = np.abs(rs.normal(marker_mu, marker_sigma, size=(size, num_features)))

    # Generate labels
    w_y = rs.normal(size=(num_features, ))
    logits = -np.linalg.norm(
        cen.reshape(1, size, 2) - cen.reshape(size, 1, 2), axis=2
    )
    threshold = np.sort(logits.reshape(-1))[-n_edge-size]
    adj = (logits >= threshold).astype(float)
    y_mean = np.diag(1. / adj.sum(axis=0)).dot(adj).dot(marker).dot(w_y)
    y_cov = np.eye(size)
    y = rs.multivariate_normal(y_mean, y_cov)
    threshold = np.median(y)
    label = (y >= threshold).astype(float)
    label[np.where(label == 0)[0]] = 0
    label = np.expand_dims(label, axis=1)

    if save:
        pickle.dump((loc, marker, label), open(os.path.join("data", filename), "wb"))
    return loc, marker, label


def generate_dataset(num_features=7, size=300, n_edge=5000, seed=0):
    loc, marker, label = generate_sample(num_features=num_features, size=size, seed=seed, save=False)
    return SyntheticDataset(2, num_features, loc, marker, label, n_edge)


def load_dataset():
    locations = ['XMin', 'XMax', 'YMin', 'YMax']
    celltype = ['Alpha', 'Beta', 'Delta']
    markers = ['CPEP', 'GCG', 'SST']
    metrics = ['Cytoplasm Intensity', '% Cytoplasm Completeness']
    scores = []

    for marker in markers:
        for metric in metrics:
            scores.append(marker+' '+metric)

    # for filename in os.listdir('data/'):
    dataset = None
    for filename in ['ABHQ115-T2D-Islet.csv', 'AFG1440-ND-Islet.csv']:
        if 'Islet.csv' in filename:
            df = pd.read_csv((os.path.join('data/', filename)))
            df = df.loc[(df['Alpha'] == 1) | (df['Beta'] == 1) | (df['Delta'] == 1)]
            df = df[locations + celltype + scores]
            marker = []
            for score in scores:
                marker.append(df[score])
            marker = np.asarray(marker).T
            if 'ND' in filename:
                label = np.zeros((marker.shape[0], 1), dtype=int)
            else:
                label = np.ones((marker.shape[0], 1), dtype=int)
            loc = np.asarray([df['XMin'], df['XMax'], df['YMin'], df['YMax']])
            if dataset is None:
                dataset = SyntheticDataset(
                    2, marker.shape[1], loc, marker, label)
            else:
                dataset.merge(SyntheticDataset(
                    2, marker.shape[1], loc, marker, label))
    return dataset


def visualize_dataset():
    def box_plot(df, filename):
        x_sz = np.max(df['XMax'])
        y_sz = np.max(df['YMax'])
        df = df.loc[(df['Alpha'] == 1) | (df['Beta'] == 1) | (df['Delta'] == 1)]
        fig, ax = plt.subplots()
        for tp, color in {'Alpha': 'y', 'Beta': 'b', 'Delta': 'r'}.items():
            df_type = df.loc[df[tp] == 1]
            cells = []
            for idx, row in df_type.iterrows():
                rect = Rectangle((row['XMin'], row['YMin']),
                                 row['XMax']-row['XMin'],
                                 row['YMax']-row['YMin'],
                                 facecolor=color, alpha=1)
                cells.append(rect)
            pc = PatchCollection(cells, match_original=True)
            ax.add_collection(pc)
        plt.xlim(0, x_sz)
        plt.ylim(0, y_sz)

        a = Patch(color='y', label='Alpha')
        b = Patch(color='b', label='Beta')
        d = Patch(color='r', label='Delta')

        ax.legend(handles=[a, b, d], loc='upper right')
        plt.savefig('plot/box/'+filename[:-4])
        plt.show()

    def scatter_plot(df, filename):
        x_sz = np.max(df['XMax'])
        y_sz = np.max(df['YMax'])
        df = df.loc[(df['Alpha'] == 1) | (df['Beta'] == 1) | (df['Delta'] == 1)]
        fig, ax = plt.subplots()
        for tp, color in {'Alpha': 'tab:orange', 'Beta': 'tab:blue', 'Delta': 'tab:red'}.items():
            df_type = df.loc[df[tp] == 1]
            x = np.asarray(df_type['XMin'] + df_type['XMax'])/2
            y = np.asarray(df_type['YMin'] + df_type['YMax'])/2
            scale = np.asarray(df_type['XMax'] + df_type['YMax'] - df_type['XMin'] - df_type['YMax'])/2
            ax.scatter(x, y, c=color, s=scale, label=tp, alpha=0.75, edgecolors='none')
        plt.xlim(0, x_sz)
        plt.ylim(0, y_sz)
        ax.legend(loc='upper right')
        plt.savefig('plot/scatter/'+filename[:-4])
        plt.show()

    locations = ['XMin', 'XMax', 'YMin', 'YMax']
    celltype = ['Alpha', 'Beta', 'Delta']

    for filename in os.listdir('data/'):
        if 'Islet.csv' in filename:
            df = pd.read_csv((os.path.join('data/', filename)))
            df = df[locations + celltype]
            box_plot(df, filename)


if __name__ == "__main__":
    visualize_dataset()
