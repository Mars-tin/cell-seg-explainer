"""
GCN
    Constructs a pytorch model for a GCN
"""
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn.conv import GCNConv


class GCN(nn.Module):
    """
    2-layer GCN:
    input -> dropout | conv+ReLU | dropout | conv -> output
    """
    def __init__(self,
                 num_features,
                 hidden_size,
                 num_classes=2,
                 dropout=0):
        super(GCN, self).__init__()

        self.conv1 = GCNConv(num_features,
                             hidden_size)
        self.conv2 = GCNConv(hidden_size,
                             num_classes)

        self.dropout = dropout
        self.activation = F.relu

    def forward(self, x, edge_index):
        x = self.activation(self.conv1(x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        return x
