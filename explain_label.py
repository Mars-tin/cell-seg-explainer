import argparse
import matplotlib.pyplot as plt
from matplotlib.patches import Patch, Rectangle, FancyArrowPatch
from matplotlib.collections import PatchCollection
from seaborn import heatmap

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


def box_plot(nodes, edges, dataset, filename):
    xmin, xmax, ymin, ymax = np.array(dataset.loc)

    x_sz = 22000
    y_sz = 12000
    '''
    x_sz = 15000
    y_sz = 15000
    '''
    fig, ax = plt.subplots()
    colors = ['y', 'limegreen', 'r']
    cells = [[], [], []]
    arrows = []

    for node in nodes:
        tp = dataset.celltype[node]
        rect = Rectangle((xmin[node], ymin[node]),
                         xmax[node]-xmin[node],
                         ymax[node]-ymin[node],
                         facecolor=colors[tp], alpha=1)
        cells[tp].append(rect)

    for edge in edges:
        tail, head, _ = edge
        x_tail = (xmin[tail] + xmax[tail])/2
        y_tail = (ymin[tail] + ymax[tail])/2
        x_head = (xmin[head] + xmax[head])/2
        y_head = (ymin[head] + ymax[head])/2
        arrow = FancyArrowPatch((x_tail, y_tail), (x_head, y_head),
                                arrowstyle="->", connectionstyle="arc3,rad=0.")
        arrows.append(arrow)

    for cell in cells:
        pc = PatchCollection(cell, match_original=True)
        ax.add_collection(pc)
    pc = PatchCollection(arrows, match_original=True)
    ax.add_collection(pc)

    plt.xlim(0, x_sz)
    plt.ylim(0, y_sz)

    a = Patch(color='y', label='Alpha')
    b = Patch(color='limegreen', label='Beta')
    d = Patch(color='r', label='Delta')
    # n = Patch(color='k', label='None')

    ax.legend(handles=[a, b, d], loc='upper right')
    plt.savefig('plot/' + filename)
    plt.show()


def view_local(islet, nodes, edges, dataset, filename):
    ran, shape = islet
    x1, x2, y1, y2 = ran

    def within(x0, y0):
        return (x1 < x0 < x2) & (y1 < y0 < y2)

    xmin, xmax, ymin, ymax = np.array(dataset.loc)
    fig, ax = plt.subplots(figsize=shape)
    colors = ['r', 'limegreen', 'b']
    cells = [[], [], []]
    arrows = []

    for node in nodes:
        tp = dataset.celltype[node]
        if not within(xmin[node], ymin[node]):
            continue
        rect = Rectangle((xmin[node], ymin[node]),
                         xmax[node]-xmin[node],
                         ymax[node]-ymin[node],
                         facecolor=colors[tp], alpha=1)
        cells[tp].append(rect)

    for edge in edges:
        tail, head, _ = edge
        x_tail = (xmin[tail] + xmax[tail])/2
        y_tail = (ymin[tail] + ymax[tail])/2
        if not within(x_tail, y_tail):
            continue
        x_head = (xmin[head] + xmax[head])/2
        y_head = (ymin[head] + ymax[head])/2
        if not within(x_head, y_head):
            continue
        arrow = FancyArrowPatch((x_tail, y_tail), (x_head, y_head),
                                arrowstyle="->", connectionstyle="arc3,rad=0.")
        arrows.append(arrow)

    for cell in cells:
        pc = PatchCollection(cell, match_original=True)
        ax.add_collection(pc)
    pc = PatchCollection(arrows, match_original=True)
    ax.add_collection(pc)

    plt.xlim(x1, x2)
    plt.ylim(y1, y2)

    a = Patch(color='r', label='Alpha')
    b = Patch(color='limegreen', label='Beta')
    d = Patch(color='b', label='Delta')

    ax.legend(handles=[a, b, d], loc='upper right')
    plt.savefig('plot/' + filename)
    plt.show()


def get_indices_kmeans(model, data, args, k=50):
    """
    Small:
    RESULT: for label=1, k = 12/35 for embedding, 40 for location
    RESULT: for label=0, k = 10/55 for embedding, 35 for location
    Medium:
    RESULT: for label=1, k = 11/40 for embedding, 40 for location
    RESULT: for label=0, k = 14/35 for embedding, 35 for location
    """

    # Perform K-Means for centers
    '''
    with torch.no_grad():
        embedding = model.encode(data.X, data.edge_index)
    x = embedding[np.where(data.y == args.label)[0]]
    ran = np.append(np.arange(2, 15), np.arange(15, 75, 5))
    find_best_k(x, ran)
    x = np.array(data.loc).T[np.where(data.y == args.label)[0]]
    ran = np.append(np.arange(2, 15), np.arange(15, 75, 5))
    find_best_k(x, ran)
    '''

    '''
    x = np.array(data.loc).T[np.where(data.y == args.label)[0]]
    x = np.array([x[:, 0:2].sum(axis=1), x[:, 2:4].sum(axis=1)]) // 2
    x = np.transpose(x)
    '''
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
    return indices


def get_indices(model, data, args, k=500):
    with torch.no_grad():
        x_embed = model.encode(data.X, data.edge_index)
        logits = model(data.X, data.edge_index)

    # Calculate entropy
    logits = logits[np.where(data.y == args.label)[0]]
    neg_entropy = torch.sum(F.softmax(logits, dim=1) * F.log_softmax(logits, dim=1), dim=1)

    # Calculate density
    kmeans = KMeans(n_clusters=k, init='k-means++')
    x_embed = x_embed[np.where(data.y == args.label)[0]]
    kmeans.fit(x_embed)
    centers = kmeans.cluster_centers_
    label = kmeans.predict(x_embed)
    centers = centers[label]
    dist_map = np.linalg.norm(x_embed - centers, axis=1)
    density = torch.tensor(1 / (1 + dist_map))

    score = density + neg_entropy
    _, indices = torch.topk(score, k=k)
    return indices


def main(parser):
    # Parameter
    args, _ = parser.parse_known_args()
    load_file = 'endo_seed=0.tar'
    data = 'real'
    model_name = args.model
    k = args.kval
    verbose = args.verbose
    epochs = 1000
    seed = 0
    threshold = 0.95
    # threshold = 0.5

    # Training
    if load_file is None:
        epochs, data, model = train(model_name, data, epochs, seed, verbose=(data == 'real'))
    else:
        epoch, data, model = load_model(model_name, data, seed, load_file)

    k = 50
    # indices = get_indices_kmeans(model, data, args, k=50)
    indices = get_indices(model, data, args, k=k)
    explainer = GNNExplainer(model, num_hops=None, epochs=epochs)

    all_nodes = []
    all_edges = []
    marker_strength = np.zeros(data.num_features)

    for node_idx in indices:
        '''
        node_idx = int(node_idx) + data.sizes[0]
        '''
        node_idx = int(node_idx)
        if node_idx > data.sizes[0]:
            continue

        '''
        if node_idx > data.sizes[0] + data.sizes[1] + data.sizes[2]:
            node_idx += data.sizes[3]
        '''

        try:
            node_feat_mask, edge_mask = explainer.explain_node(
                node_idx, data.X, data.edge_index)
            if verbose:
                print("Location:", data.loc[:, node_idx])
                print("Marker mask:", node_feat_mask)
            G = explainer.visualize_subgraph(node_idx, data, edge_mask, pos=None,
                                             y=data.y, show=False, verbose=verbose)

            for source, target, graph_data in G.edges(data=True):
                if graph_data['att'] > threshold:
                    all_edges.append([source, target, graph_data['att']])
            all_nodes.extend(list(G.nodes))
            marker_strength += np.array(node_feat_mask)

            # scatter_plot(nodes, data, 'local_{}'.format(node_idx))
            # box_plot(nodes, data, 'local_{}'.format(node_idx))

        except TypeError:
            print("Node {} is not explainable".format(node_idx))
            continue

    # Global view
    # scatter_plot(all_nodes, data, 'scatter_{}'.format(k))
    box_plot(all_nodes, all_edges, data, 'box_{}'.format(k))

    # Local view
    islets = [[(11750, 12750, 6250, 7500), (10, 12)]]
    for idx, islet in enumerate(islets):
        view_local(islet, all_nodes, all_edges, data, 'islet_{}_{}'.format(idx, k))

    # Feature View
    marker_strength /= np.sum(marker_strength)
    print("Total marker strength:", marker_strength)
    plt.figure(figsize=(12, 2))
    heatmap(np.expand_dims(marker_strength, axis=0), square=False, cmap='Blues',
            vmax=1/2, vmin=0, xticklabels=[], yticklabels=[], annot=True)
    plt.savefig('plot/' + 'markers')
    plt.show()


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
        default=1,
        help='Index of label to explain'
    )
    parser.add_argument(
        '-k', '--kval',
        type=int,
        default=50,
        help='Print extra information'
    )
    parser.add_argument(
        '-v', '--verbose',
        type=bool,
        default=False,
        help='Print extra information'
    )
    main(parser)
