import matplotlib.pyplot as plt
import networkx as nx

import torch
from torch_geometric.data import Data
from torch_geometric.utils import k_hop_subgraph, to_networkx
from torch_geometric.nn.models import GNNExplainer

from train import train


if __name__ == "__main__":
    # Parameter
    verbose = True
    node_idx = 1
    model_name = 'gcn'
    epochs = 1000
    seed = 6

    # Training
    epochs, data, model = train(model_name, epochs, seed)

    # Explaining
    print("Node: ", node_idx, "; Label:", data.y[node_idx].item())
    explainer = GNNExplainer(model, epochs=epochs)
    node_feat_mask, edge_mask = explainer.explain_node(node_idx, data.X, data.edge_index)

    # Verbose
    if verbose:
        num_hops = 2
        subset, edge_index, _, hard_edge_mask = k_hop_subgraph(node_idx, num_hops, data.edge_index, relabel_nodes=True)
        hard_mask = edge_mask[hard_edge_mask]
        y = torch.zeros(edge_index.max().item() + 1)
        graph_data = Data(edge_index=edge_index, att=hard_mask, y=y, num_nodes=y.size(0))
        G = to_networkx(graph_data, node_attrs=['y'], edge_attrs=['att'])
        mapping = {k: i for k, i in enumerate(subset.tolist())}
        G = nx.relabel_nodes(G, mapping)
        print("Related nodes:", G.nodes)
        print("Related edges:", G.edges)
        for node in G.nodes:
            print("Node:", node, "; Marker:", data.X[node])

    # Visualization
    ax, g = explainer.visualize_subgraph(node_idx, data.edge_index, edge_mask, y=data.y)
    plt.show()
