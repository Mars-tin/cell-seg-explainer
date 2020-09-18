import argparse
import matplotlib.pyplot as plt
from matplotlib.patches import Patch, Rectangle
from matplotlib.collections import PatchCollection

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score

from model.gnnexplainer import GNNExplainer
from train import *


def find_best_k(x, ran):
    scores = []
    for num_cluster in ran:
        kmeans = KMeans(n_clusters=num_cluster, init='k-means++')
        kmeans.fit(x)
        wss = kmeans.inertia_
        sil = silhouette_score(x, kmeans.labels_, metric='euclidean')
        db = davies_bouldin_score(x, kmeans.labels_)
        scores.append([wss, sil, db])
        print(num_cluster, wss, sil, db)

    fig, axs = plt.subplots(3)
    for i, metric in enumerate(['Inertia', 'Silhouette Value', 'Davies-Bouldin']):
        axs[i].plot(ran, np.array(scores)[:, i], marker='o', label=metric)
        axs[i].legend(loc='upper right')
    plt.show()


def scatter_plot(nodes, dataset, filename):
    xmin, xmax, ymin, ymax = np.array(dataset.loc)
    x = np.asarray(xmin + xmax) / 2
    y = np.asarray(ymin + ymax) / 2
    scale = np.asarray(xmax + ymax - xmin - ymin) / 2
    x_sz = 22000
    y_sz = 12000

    fig, ax = plt.subplots()

    types = ['Alpha', 'Beta', 'Delta']
    cells = {'Alpha': [], 'Beta': [],  'Delta': []}
    for node in nodes:
        tp = dataset.celltype[node]
        cells[types[tp]].append(node)

    for tp, color in {'Alpha': 'tab:orange', 'Beta': 'tab:blue', 'Delta': 'tab:red'}.items():
        x_tp = x[cells[tp]]
        y_tp = y[cells[tp]]
        scale_tp = scale[cells[tp]]
        ax.scatter(x_tp, y_tp, c=color, s=scale_tp, label=tp, alpha=0.75, edgecolors='none')

    plt.xlim(0, x_sz)
    plt.ylim(0, y_sz)

    ax.legend(loc='upper right')
    plt.savefig('plot/' + filename)
    plt.show()


def box_plot(nodes, dataset, filename):
    xmin, xmax, ymin, ymax = np.array(dataset.loc)
    x_sz = 22000
    y_sz = 12000

    fig, ax = plt.subplots()
    colors = ['y', 'b', 'r']
    cells = [[], [], []]

    for node in nodes:
        tp = dataset.celltype[node]
        rect = Rectangle((xmin[node], ymin[node]),
                         xmax[node]-xmin[node],
                         ymax[node]-ymin[node],
                         facecolor=colors[tp], alpha=1)
        cells[tp].append(rect)

    for cell in cells:
        pc = PatchCollection(cell, match_original=True)
        ax.add_collection(pc)

    plt.xlim(0, x_sz)
    plt.ylim(0, y_sz)

    a = Patch(color='y', label='Alpha')
    b = Patch(color='b', label='Beta')
    d = Patch(color='r', label='Delta')
    # n = Patch(color='k', label='None')

    ax.legend(handles=[a, b, d], loc='upper right')
    plt.savefig('plot/' + filename)
    plt.show()


def view_local(islet, nodes, dataset, filename):
    ran, shape = islet
    xmin, xmax, ymin, ymax = np.array(dataset.loc)
    fig, ax = plt.subplots(figsize=shape)
    colors = ['y', 'b', 'r']
    cells = [[], [], []]

    for node in nodes:
        tp = dataset.celltype[node]
        rect = Rectangle((xmin[node], ymin[node]),
                         xmax[node]-xmin[node],
                         ymax[node]-ymin[node],
                         facecolor=colors[tp], alpha=1)
        cells[tp].append(rect)

    for cell in cells:
        pc = PatchCollection(cell, match_original=True)
        ax.add_collection(pc)

    x1, x2, y1, y2 = ran
    plt.xlim(x1, x2)
    plt.ylim(y1, y2)

    a = Patch(color='y', label='Alpha')
    b = Patch(color='b', label='Beta')
    d = Patch(color='r', label='Delta')

    ax.legend(handles=[a, b, d], loc='upper right')
    plt.savefig('plot/' + filename)
    plt.show()


def main(parser):
    # Parameter
    args, _ = parser.parse_known_args()
    load_file = 'model_seed=0.tar'
    data = 'real'
    model_name = args.model
    verbose = args.verbose
    epochs = 1000
    seed = 0

    # Training
    if load_file is None:
        epochs, data, model = train(model_name, data, epochs, seed, verbose=(data == 'real'))
    else:
        epoch, data, model = load_model(model_name, data, seed, load_file)

    # Perform K-Means for centers
    '''
    with torch.no_grad():
        embedding = model.encode(data.X, data.edge_index)
    embedding = embedding[np.where(data.y == args.label)[0]]
    ran = np.append(np.arange(2, 15), np.arange(15, 75, 5))
    find_best_k(x, ran)
    x = np.array(data.loc).T[np.where(data.y == args.label)[0]]
    ran = np.append(np.arange(2, 15), np.arange(15, 75, 5))
    find_best_k(x, ran)
    '''

    # RESULT: k = 6 for embedding, 40 for location
    k = 50
    verbose = False

    # x = np.array(data.loc).T[np.where(data.y == args.label)[0]]
    with torch.no_grad():
        x = model.encode(data.X, data.edge_index)
    x = x[np.where(data.y == args.label)[0]]

    kmeans = KMeans(n_clusters=k, init='k-means++')
    kmeans.fit(x)
    centers = kmeans.cluster_centers_

    indices = []
    for center in centers:
        dist_map = np.linalg.norm(x - center, axis=1)
        dist_map[indices] = np.infty
        idx = np.argmin(dist_map)
        indices.append(idx)

    explainer = GNNExplainer(model, epochs=epochs)

    all_nodes = []
    marker_strength = np.zeros(data.num_features)

    for node_idx in indices:
        node_idx = int(node_idx) + data.sizes[0]

        try:
            node_feat_mask, edge_mask = explainer.explain_node(
                node_idx, data.X, data.edge_index)
            if verbose:
                print("Marker mask:", node_feat_mask)
            nodes = explainer.visualize_subgraph(
                node_idx, data, edge_mask,
                y=data.y, show=False, verbose=verbose)

            all_nodes.extend(list(nodes))
            marker_strength += np.array(node_feat_mask)

            # scatter_plot(nodes, data, 'local_{}'.format(node_idx))
            # box_plot(nodes, data, 'local_{}'.format(node_idx))

        except TypeError:
            print("Node {} is not explainable".format(node_idx))
            continue

    scatter_plot(all_nodes, data, 'scatter_{}'.format(k))
    box_plot(all_nodes, data, 'box_{}'.format(k))

    '''
    islets = [
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
    '''
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
    for idx, islet in enumerate(islets):
        view_local(islet, all_nodes, data, 'islet_{}_{}'.format(idx, k))
    print("Total marker strength:", marker_strength)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-m', '--model',
        type=str,
        default='gcn',
        help='Name of the model'
    )
    parser.add_argument(
        '-l', '--label',
        type=int,
        default=0,
        help='Index of label to explain'
    )
    parser.add_argument(
        '-k', '--kval',
        type=int,
        default=6,
        help='Print extra information'
    )
    parser.add_argument(
        '-v', '--verbose',
        type=bool,
        default=True,
        help='Print extra information'
    )
    main(parser)
