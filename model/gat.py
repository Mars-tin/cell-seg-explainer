"""
GAT
    Constructs a pytorch model for a GAT
"""
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv


class GAT(nn.Module):
    """
    2-layer GAT:
    input -> dropout | conv+ReLU | dropout | conv -> output
    """
    def __init__(self,
                 num_features,
                 hidden_size,
                 num_targets=1,
                 num_heads=8,
                 dropout=0):
        super(GAT, self).__init__()

        self.conv1 = GATConv(num_features,
                             hidden_size,
                             heads=num_heads,
                             dropout=dropout)
        self.conv2 = GATConv(hidden_size * num_heads,
                             num_targets,
                             heads=1,
                             dropout=dropout)

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
