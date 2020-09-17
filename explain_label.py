import argparse
import networkx as nx

from sklearn.cluster import KMeans

from torch_geometric.data import Data
from torch_geometric.utils import k_hop_subgraph, to_networkx
from torch_geometric.nn import MessagePassing
from torch_geometric.nn.models import GNNExplainer

from train import *


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

    with torch.no_grad():
        embedding = model.encode(data.X, data.edge_index)
    num_hops = 0
    for module in model.modules():
        if isinstance(module, MessagePassing):
            num_hops += 1

    # Perform K-Means for centers
    for num_cluster in np.arange(10, 100, 10):
        kmeans = KMeans(n_clusters=num_cluster, init='k-means++')
        kmeans.fit(embedding)
        wss = kmeans.inertia_
        print(num_cluster, wss)

    centers = kmeans.cluster_centers_
    indices = []
    for center in centers:
        dist_map = np.linalg.norm(embedding - center, axis=1)
        dist_map[indices] = np.infty
        idx = np.argmin(dist_map)
        indices.append(idx)

    # Explain
    explainer = GNNExplainer(model, epochs=epochs)

    for node_idx in indices:
        node_feat_mask, edge_mask = explainer.explain_node(node_idx, data.X, data.edge_index)
        subset, edge_index, _, hard_edge_mask = k_hop_subgraph(
            node_idx, num_hops, data.edge_index, relabel_nodes=True)
        hard_mask = edge_mask[hard_edge_mask]
        y = data.y[subset].to(torch.float) / data.y.max().item()
        graph_data = Data(edge_index=edge_index, att=hard_mask, y=y, num_nodes=y.size(0))
        try:
            G = to_networkx(graph_data, node_attrs=['y'], edge_attrs=['att'])
            mapping = {k: i for k, i in enumerate(subset.tolist())}
            G = nx.relabel_nodes(G, mapping)
            if verbose:
                print("Node: ", node_idx, "; Label:", data.y[node_idx].item())
                print("Related nodes:", G.nodes)
                print("Related edges:", G.edges)
                print("Marker mask:", node_feat_mask)
                for node in G.nodes:
                    print("Node:", node, "; Label:", data.y[node].item(), "; Marker:", data.X[node])
        except TypeError:
            print("Unexplainable node {}".format(node_idx))
            pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-m', '--model',
        type=str,
        default='gcn',
        help='Name of the model'
    )
    parser.add_argument(
        '-v', '--verbose',
        type=bool,
        default=False,
        help='Print extra information'
    )
    main(parser)
