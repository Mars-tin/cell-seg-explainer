import os
import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection
from matplotlib.patches import Patch, Rectangle

import torch
from torch.utils.data import Dataset
import torch_geometric.transforms as T


class SyntheticDataset(Dataset):

    def __init__(self, num_classes, num_features, loc, marker, label, n_edge=None, **kwargs):
        super().__init__()
        self.num_classes = num_classes
        self.num_features = num_features

        self.X = torch.from_numpy(marker).float()
        self.y = torch.from_numpy(label).long()
        self.loc = torch.from_numpy(loc)
        self.celltype = torch.from_numpy(kwargs.get('celltype')).int()

        n = self.__len__()
        self.sizes = [n]
        self.samples = [kwargs.get('sample')]

        if n_edge is None:
            n_edge = 10 * n
        x = np.mean(loc[0:2], axis=0)
        y = np.mean(loc[2:4], axis=0)
        center = np.stack((x, y)).transpose()
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
        self.loc = torch.cat((self.loc, dataset.loc), 1)
        self.celltype = torch.cat((self.celltype, dataset.celltype), 0)
        self.sizes.append(dataset.__len__())


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


def generate_dataset(num_features=7, size=300, n_edge=None, seed=0):
    loc, marker, label = generate_sample(num_features=num_features, size=size, seed=seed, save=False)
    return SyntheticDataset(2, num_features, loc, marker, label, n_edge)


def load_dataset():
    locations = ['XMin', 'XMax', 'YMin', 'YMax']
    celltype = ['Alpha', 'Beta', 'Delta']
    markers = ['CPEP', 'GCG', 'SST']
    metrics = ['Cytoplasm Intensity', '% Cytoplasm Completeness']
    # metrics = ['Cytoplasm Intensity']

    # files = ['ABHQ115-T2D-Islet.csv', 'AFG1440-ND-Islet.csv']
    files = ['ABHQ115-T2D-Islet.csv', 'AFG1440-ND-Islet.csv', 'ABIC495-T2D-Islet.csv']
    # files = ['ABHQ115-T2D-Islet.csv', 'AFG1440-ND-Islet.csv', 'ABIC495-T2D-Islet.csv',
    #          'AFES372-ND-Islet.csv', 'ADLE098-T2D-Islet.csv']

    scores = []

    for marker in markers:
        for metric in metrics:
            scores.append(marker+' '+metric)

    # for filename in os.listdir('data/'):
    dataset = None
    for filename in files:
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
            tp = np.asarray([df['Beta'] - df['Delta']]).T
            if dataset is None:
                dataset = SyntheticDataset(
                    2, marker.shape[1], loc, marker, label, celltype=tp, sample=filename[:7])
            else:
                dataset.merge(SyntheticDataset(
                    2, marker.shape[1], loc, marker, label, celltype=tp, sample=filename[:7]))
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
            scatter_plot(df, filename)


def view_islet(filename, ran, id, shape=(10, 15)):
    locations = ['XMin', 'XMax', 'YMin', 'YMax']
    celltype = ['Alpha', 'Beta', 'Delta']

    df = pd.read_csv((os.path.join('data/', filename)))
    df = df[locations + celltype]

    df = df.loc[(df['Alpha'] == 1) | (df['Beta'] == 1) | (df['Delta'] == 1)]
    fig, ax = plt.subplots(figsize=shape)
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

    x1, x2, y1, y2 = ran
    plt.xlim(x1, x2)
    plt.ylim(y1, y2)
    a = Patch(color='y', label='Alpha')
    b = Patch(color='b', label='Beta')
    d = Patch(color='r', label='Delta')
    ax.legend(handles=[a, b, d], loc='upper right')
    plt.savefig('plot/islet_{}_ran={}'.format(id, ran))
    plt.show()


if __name__ == "__main__":
    islets = [
        [(8400, 9800, 3800, 5800), (10, 15)],
        [(4500, 5200, 2700, 4200), (10, 15)],
        [(2500, 3100, 3300, 4000), (10, 12)],
        [(9500, 10200, 6400, 7700), (10, 15)],
        [(3600, 4400, 1500, 3000), (10, 15)],
        [(7300, 8100, 6000, 6800), (10, 10)],
        [(8200, 9300, 8300, 9300), (10, 8)],
        [(6000, 6800, 200, 1700), (10, 15)],
        [(6000, 6800, 1800, 2600), (10, 10)],
        [(3100, 4000, 6500, 7400), (10, 10)],
        [(400, 1200, 5000, 5800), (10, 10)]
    ]
    filename = 'ABHQ115-T2D-Islet.csv'
    for idx in range(10):
        view_islet(filename, islets[idx][0], idx, shape=islets[idx][1])

    islets = [
        [(900, 1300, 500, 1000), (10, 10)],
        [(1400, 2200, 1500, 2100), (10, 10)],
        [(2300, 2900, 800, 1800), (10, 15)],
        [(3600, 4700, 1800, 2600), (10, 8)],
        [(5200, 6100, 5700, 6400), (10, 8)],
        [(8300, 9100, 6100, 6900), (10, 10)],
        [(15000, 16100, 6200, 7600), (10, 12)],
        [(13000, 14300, 8400, 9300), (10, 8)],
        [(13200, 15200, 2500, 3300), (10, 6)],
        [(13500, 14100, 3500, 3900), (10, 10)]
    ]
    filename = 'AFG1440-ND-Islet.csv'
    for idx in range(10):
        view_islet(filename, islets[idx][0], idx, shape=islets[idx][1])
