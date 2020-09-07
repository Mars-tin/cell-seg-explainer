"""
MLP
    Constructs a pytorch model for a MLP
"""
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    """
    2-layer MLP:
    input -> dropout | fc+ReLU | dropout | fc -> output
    """
    def __init__(self,
                 num_features,
                 hidden_size,
                 num_targets=1,
                 dropout=0):
        super(MLP, self).__init__()

        self.fc1 = nn.Linear(num_features,
                             hidden_size)
        self.fc2 = nn.Linear(hidden_size,
                             num_targets)

        self.dropout = dropout
        self.activation = F.relu

    def forward(self, x, edge_index):
        x = self.activation(self.fc1(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.fc2(x)
        return x
