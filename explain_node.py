import argparse
from math import sqrt
import matplotlib.pyplot as plt
import networkx as nx

import torch
from torch_geometric.data import Data
from torch_geometric.utils import k_hop_subgraph, to_networkx
from torch_geometric.nn import MessagePassing
from torch_geometric.nn.models import GNNExplainer

from train import train


def main(parser):
    # Parameter
    args, _ = parser.parse_known_args()
    model_name = args.model
    node_idx = args.node
    verbose = args.verbose
    epochs = 1000
    seed = 1

    # Training
    epochs, data, model = train(model_name, epochs, seed)
    num_hops = 0
    for module in model.modules():
        if isinstance(module, MessagePassing):
            num_hops += 1

    # Explaining
    explainer = GNNExplainer(model, epochs=epochs)
    node_feat_mask, edge_mask = explainer.explain_node(node_idx, data.X, data.edge_index)
    subset, edge_index, _, hard_edge_mask = k_hop_subgraph(node_idx, num_hops, data.edge_index, relabel_nodes=True)
    hard_mask = edge_mask[hard_edge_mask]
    y = data.y[subset].to(torch.float) / data.y.max().item()
    graph_data = Data(edge_index=edge_index, att=hard_mask, y=y, num_nodes=y.size(0))
    G = to_networkx(graph_data, node_attrs=['y'], edge_attrs=['att'])
    mapping = {k: i for k, i in enumerate(subset.tolist())}
    G = nx.relabel_nodes(G, mapping)
    if verbose:
        print("Node: ", node_idx, "; Label:", data.y[node_idx].item())
        print("Related nodes:", G.nodes)
        print("Related edges:", G.edges)
        print("Marker importance:", node_feat_mask)
        for node in G.nodes:
            print("Node:", node, "; Label:", data.y[node].item(), "; Marker:", data.X[node])

    # Visualization
    pos = nx.spring_layout(G)
    ax = plt.gca()
    for source, target, graph_data in G.edges(data=True):
        ax.annotate(
            '', xy=pos[target], xycoords='data', xytext=pos[source],
            textcoords='data', arrowprops=dict(
                arrowstyle="->",
                alpha=max(graph_data['att'], 0.1),
                shrinkA=sqrt(1000) / 2.0,
                shrinkB=sqrt(1000) / 2.0,
                connectionstyle="arc3,rad=0.1",
            ))
    nx.draw_networkx_nodes(G, pos, node_color=y.flatten(), cmap='Set3')
    nx.draw_networkx_labels(G, pos)
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
        '-n', '--node',
        type=int,
        default=1,
        help='Index of node to explain'
    )
    parser.add_argument(
        '-v', '--verbose',
        type=bool,
        default=True,
        help='Print extra information'
    )
    main(parser)
