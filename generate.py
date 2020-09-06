import argparse
import os
import numpy as np
import pickle


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
    cen = rs.uniform(0, fig_size, size=(2, size))
    rad = np.abs(rs.normal(radius_mu, radius_sigma, size=(2, size)))
    loc = np.array([cen[0]-rad[0], cen[0]+rad[0], cen[1]-rad[1], cen[1]+rad[1]])
    marker = np.abs(rs.normal(marker_mu, marker_sigma, size=(7,size)))
    label = rs.choice(labels, size)

    if save:
        pickle.dump((loc, marker, label), open(os.path.join("data", filename), "wb"))
    return loc, marker, label


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-n', '--size',
        type=int,
        default=200,
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
