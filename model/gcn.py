"""
GCN
    Constructs a pytorch model for a GCN
"""
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv


class GCN(nn.Module):
    """
    2-layer GCN:
    input -> dropout | conv+ReLU | dropout | conv -> output
    """
    def __init__(self,
                 num_features,
                 hidden_size,
                 num_targets=1,
                 dropout=0):
        super(GCN, self).__init__()

        self.conv1 = GCNConv(num_features,
                             hidden_size)
        self.conv2 = GCNConv(hidden_size,
                             num_targets)

        self.dropout = dropout
        self.activation = F.relu

    def forward(self, data):
        x = data.X
        edge_index = data.edge_index
        """
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.activation(self.conv1(x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        """
        x = self.activation(self.conv1(x, edge_index))
        x = self.conv2(x, edge_index)
        return x.flatten()
