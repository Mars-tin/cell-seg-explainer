import argparse
from model.gnnexplainer import GNNExplainer

from train import *


def main(parser):
    # Parameter
    args, _ = parser.parse_known_args()
    load_file = 'model_seed=0.tar'
    data = 'real'
    model_name = args.model
    node_idx = args.node
    verbose = args.verbose
    epochs = 1000
    seed = 0

    # Training
    if load_file is None:
        epochs, data, model = train(model_name, data, epochs, seed, verbose=(data == 'real'))
    else:
        epoch, data, model = load_model(model_name, data, seed, load_file)

    explainer = GNNExplainer(model, epochs=epochs)

    try:
        node_feat_mask, edge_mask = explainer.explain_node(node_idx, data.X, data.edge_index)
        if verbose:
            print("Marker mask:", node_feat_mask)
        explainer.visualize_subgraph(node_idx, data, edge_mask, y=data.y, verbose=verbose)
    except TypeError:
        print("Node {} is not explainable".format(node_idx))


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
        default=2,
        help='Index of node to explain'
    )
    parser.add_argument(
        '-v', '--verbose',
        type=bool,
        default=True,
        help='Print extra information'
    )
    main(parser)
