"""
GAT
    Constructs a pytorch model for a GAT
"""
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn.conv import GATConv


class GAT(nn.Module):
    """
    2-layer GAT:
    input -> dropout | conv+ReLU | dropout | conv -> output
    """
    def __init__(self,
                 num_features,
                 hidden_size,
                 num_classes=2,
                 num_heads=8,
                 dropout=0):
        super(GAT, self).__init__()

        self.conv1 = GATConv(num_features,
                             hidden_size,
                             heads=num_heads,
                             dropout=dropout)
        self.conv2 = GATConv(hidden_size * num_heads,
                             num_classes,
                             heads=1,
                             dropout=dropout)

        self.dropout = dropout
        self.activation = F.relu

    def forward(self, x, edge_index):
        x = self.activation(self.conv1(x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        return x
