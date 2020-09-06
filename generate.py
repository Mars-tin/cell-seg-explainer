import argparse
import os
import numpy as np
import pickle
import torch

from torch.utils.data import Dataset


def generate_sample(size=200, seed=0, save=True):
    """
    Cell information:
    * Location: square box, x, y, r1, r2
    * Marker: a vector of size 7
    * Label: -: {n1, n2, n3, n4} & +: {p1, p2, p3, p4, p5}
    """
    # Parameters
    filename = 'cell_sample_size={}_seed={}.pkl'.format(size, seed)
    fig_size = 500
    marker_mu = 3
    marker_sigma = 1
    radius_mu = 10
    radius_sigma = 1
    labels = [-1, -2, -3, -4, 1, 2, 3, 4, 5]

    # Randomization
    rs = np.random.RandomState(seed=seed)
    cen = rs.uniform(0, fig_size, size=(size, 2))
    rad = np.abs(rs.normal(radius_mu, radius_sigma, size=(size, 2)))
    loc = np.array([cen[:, 0]-rad[:, 0], cen[:, 0]+rad[:, 0], cen[:, 1]-rad[:, 1], cen[:, 1]+rad[:, 1]])
    marker = np.abs(rs.normal(marker_mu, marker_sigma, size=(size, 7)))
    label = rs.choice(labels, size)

    if save:
        pickle.dump((loc, marker, label), open(os.path.join("data", filename), "wb"))
    return loc, marker, label


class SyntheticDataset(Dataset):

    def __init__(self, loc, marker, label, n_edge):
        super().__init__()
        self.X = torch.from_numpy(marker).float()
        self.y = torch.from_numpy(label).float()
        n = self.__len__()

        x = np.mean(loc[0:2], axis=0)
        y = np.mean(loc[2:4], axis=0)
        center = np.concatenate((x, y))
        logits = -np.linalg.norm(
            center.reshape(1, n, 2) - center.reshape(n, 1, 2), axis=2
        )
        threshold = np.sort(logits.reshape(-1))[-n_edge]
        self.adj = (logits >= threshold).astype(float)

        self.edge_index = torch.tensor(np.array(list(self.adj.nonzero())))

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


def generate_dataset(size=300, n_edge=5000, seed=0):
    loc, marker, label = generate_sample(size=size, seed=seed, save=False)
    return SyntheticDataset(loc, marker, label, n_edge)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-n', '--size',
        type=int,
        default=300,
        help='number of cells'
    )
    parser.add_argument(
        '-s', '--seed',
        type=int,
        default=0,
        help='seed for random generation'
    )
    parser.add_argument(
        '-o', '--save',
        type=bool,
        default=True,
        help='Save the output or not'
    )
    args, _ = parser.parse_known_args()
    generate_sample(args.size, args.seed)
