from math import sqrt
from typing import Optional

import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import networkx as nx

from torch_geometric.nn import MessagePassing
from torch_geometric.data import Data
from torch_geometric.utils import k_hop_subgraph, to_networkx

EPS = 1e-15


class GNNExplainer(torch.nn.Module):

    coeffs = {
        'edge_size': 0.005,
        'node_feat_size': 1.0,
        'edge_ent': 1.0,
        'node_feat_ent': 0.1,
    }

    def __init__(self, model, epochs: int = 100, lr: float = 0.01):
        super(GNNExplainer, self).__init__()
        self.model = model
        self.epochs = epochs
        self.lr = lr
        self.num_hops = self.__num_hops__()

    def __set_masks__(self, x, edge_index, init="normal"):
        (N, F), E = x.size(), edge_index.size(1)

        std = 0.1
        self.node_feat_mask = torch.nn.Parameter(torch.randn(F) * std, requires_grad=True)

        std = torch.nn.init.calculate_gain('relu') * sqrt(2.0 / (2 * N))
        self.edge_mask = torch.nn.Parameter(torch.randn(E) * std, requires_grad=True)

        for module in self.model.modules():
            if isinstance(module, MessagePassing):
                module.__explain__ = True
                module.__edge_mask__ = self.edge_mask

    def __clear_masks__(self):
        for module in self.model.modules():
            if isinstance(module, MessagePassing):
                module.__explain__ = False
                module.__edge_mask__ = None
        self.node_feat_masks = None
        self.edge_mask = None

    def __num_hops__(self):
        num_hops = 0
        for module in self.model.modules():
            if isinstance(module, MessagePassing):
                num_hops += 1
        return num_hops

    def __flow__(self):
        for module in self.model.modules():
            if isinstance(module, MessagePassing):
                return module.flow
        return 'source_to_target'

    def __subgraph__(self, node_idx, x, edge_index, **kwargs):
        num_nodes, num_edges = x.size(0), edge_index.size(1)

        if node_idx is not None:
            subset, edge_index, mapping, edge_mask = k_hop_subgraph(
                node_idx, self.num_hops, edge_index, relabel_nodes=True,
                num_nodes=num_nodes, flow=self.__flow__())
            x = x[subset]
            for key, item in kwargs.items():
                if torch.is_tensor(item) and item.size(0) == num_nodes:
                    item = item[subset]
                elif torch.is_tensor(item) and item.size(0) == num_edges:
                    item = item[edge_mask]
                kwargs[key] = item
        else:
            x = x
            edge_index = edge_index
            row, col = edge_index
            edge_mask = row.new_empty(row.size(0), dtype=torch.bool)
            edge_mask[:] = True
            mapping = None

        return x, edge_index, mapping, edge_mask, kwargs

    def __loss__(self, node_idx, log_logits, pred_label):
        loss = -log_logits[node_idx, pred_label[node_idx]]

        m = self.edge_mask.sigmoid()
        loss = loss + self.coeffs['edge_size'] * m.sum()
        ent = -m * torch.log(m + EPS) - (1 - m) * torch.log(1 - m + EPS)
        loss = loss + self.coeffs['edge_ent'] * ent.mean()

        m = self.node_feat_mask.sigmoid()
        loss = loss + self.coeffs['node_feat_size'] * m.sum()
        ent = -m * torch.log(m + EPS) - (1 - m) * torch.log(1 - m + EPS)
        loss = loss + self.coeffs['node_feat_ent'] * ent.mean()

        return loss

    def __graph_loss__(self, log_logits, pred_label):
        loss = -torch.log(log_logits[0, pred_label])
        m = self.edge_mask.sigmoid()
        loss = loss + self.coeffs['edge_size'] * m.sum()
        ent = -m * torch.log(m + EPS) - (1 - m) * torch.log(1 - m + EPS)
        loss = loss + self.coeffs['edge_ent'] * ent.mean()

        return loss

    def explain_node(self, node_idx, x, edge_index, **kwargs):

        self.model.eval()
        self.__clear_masks__()

        num_edges = edge_index.size(1)

        # Only operate on a k-hop subgraph around `node_idx`.
        x, edge_index, mapping, hard_edge_mask, kwargs = self.__subgraph__(
            node_idx, x, edge_index, **kwargs)

        # Get the initial prediction.
        with torch.no_grad():
            logits = self.model(x=x, edge_index=edge_index, **kwargs)
            pred_label = F.log_softmax(logits, dim=1).argmax(dim=-1)

        self.__set_masks__(x, edge_index)
        self.to(x.device)

        optimizer = torch.optim.Adam([self.node_feat_mask, self.edge_mask], lr=self.lr)

        for epoch in range(1, self.epochs + 1):
            optimizer.zero_grad()
            h = x * self.node_feat_mask.view(1, -1).sigmoid()
            logits = self.model(x=h, edge_index=edge_index, **kwargs)
            loss = self.__loss__(mapping, F.log_softmax(logits, dim=1), pred_label)
            loss.backward()
            optimizer.step()

        node_feat_mask = self.node_feat_mask.detach().sigmoid()
        edge_mask = self.edge_mask.new_zeros(num_edges)
        edge_mask[hard_edge_mask] = self.edge_mask.detach().sigmoid()

        self.__clear_masks__()

        return node_feat_mask, edge_mask

    def explain_graph(self, x, edge_index, **kwargs):

        self.model.eval()
        self.__clear_masks__()

        # Only operate on a k-hop subgraph around `node_idx`.
        x, edge_index, _, hard_edge_mask, kwargs = self.__subgraph__(
            node_idx=None, x=x, edge_index=edge_index, **kwargs)

        # Get the initial prediction.
        with torch.no_grad():
            logits = self.model(x=x, edge_index=edge_index, **kwargs)
            pred_label = F.log_softmax(logits, dim=1).argmax(dim=-1)

        self.__set_masks__(x, edge_index)
        self.to(x.device)

        optimizer = torch.optim.Adam([self.edge_mask], lr=self.lr)

        epoch_losses = []
        for epoch in range(1, self.epochs + 1):
            epoch_loss = 0
            optimizer.zero_grad()

            logits = self.model(x=x, edge_index=edge_index, **kwargs)
            pred = F.log_softmax(logits, dim=1).argmax(dim=-1)

            loss = self.__graph_loss__(pred, pred_label)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.detach().item()
            epoch_losses.append(epoch_loss)

        edge_mask = self.edge_mask.detach().sigmoid()
        self.__clear_masks__()

        return edge_mask, epoch_losses

    def visualize_subgraph(self, node_idx, dataset, edge_mask, y=None,
                           save=False, verbose=True, threshold=None, **kwargs):

        edge_index = dataset.edge_index

        assert edge_mask.size(0) == edge_index.size(1)

        if threshold is not None:
            print('Edge Threshold:', threshold)
            edge_mask = torch.tensor(edge_mask >= threshold, dtype=torch.float)

        if node_idx is not None:
            # Only operate on a k-hop subgraph around `node_idx`.
            subset, edge_index, _, hard_edge_mask = k_hop_subgraph(
                node_idx, self.num_hops, edge_index, relabel_nodes=True,
                num_nodes=None, flow=self.__flow__())
            edge_mask = edge_mask[hard_edge_mask]

        else:
            subset = []
            for index, mask in enumerate(edge_mask):
                node_a = edge_index[0, index]
                node_b = edge_index[1, index]
                if node_a not in subset:
                    subset.append(node_a.cpu().item())
                if node_b not in subset:
                    subset.append(node_b.cpu().item())

        if y is None:
            y = torch.zeros(edge_index.max().item() + 1, device=edge_index.device)
        else:
            y = y[subset].to(torch.float) / y.max().item()

        data = Data(edge_index=edge_index, att=edge_mask, y=y,
                    num_nodes=y.size(0)).to('cpu')

        G = to_networkx(data, node_attrs=['y'], edge_attrs=['att'])
        mapping = {k: i for k, i in enumerate(subset.tolist())}
        G = nx.relabel_nodes(G, mapping)

        kwargs['with_labels'] = kwargs.get('with_labels') or True
        kwargs['font_size'] = kwargs.get('font_size') or 10
        kwargs['node_size'] = kwargs.get('node_size') or 200
        kwargs['cmap'] = kwargs.get('cmap') or 'Set3'

        if verbose:
            print("Node: ", node_idx, "; Label:", data.y[node_idx].item())
            print("Related nodes:", G.nodes)
            print("Related edges:", G.edges)
            for node in G.nodes:
                print("Node:", node, "; Label:", dataset.y[node].item(), "; Marker:", dataset.X[node])

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

        nx.draw_networkx_nodes(G, pos, node_color=y.flatten(), **kwargs)
        nx.draw_networkx_labels(G, pos, **kwargs)

        if save:
            plt.savefig('plot/sample')
        plt.show()

        return plt

    def __repr__(self):
        return f'{self.__class__.__name__}()'
