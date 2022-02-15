import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv


class GCN(nn.Module):

    def __init__(self, num_nodes, embedding_size, hidden_size, num_classes, weights):

        super(GCN, self).__init__()

        self.embedding = nn.Embedding(num_nodes, embedding_dim=embedding_size)
        # self.embedding.weight.requires_grad = False
        self.embedding.weight = nn.Parameter(torch.from_numpy(weights).float(), requires_grad=False)

        self.conv1 = GCNConv(embedding_size, num_classes, bias=False)
        # self.conv2 = GCNConv(hidden_size, num_classes)
        self.ce = nn.CrossEntropyLoss()
        self.ce2 = nn.CrossEntropyLoss(reduction='none')
        self.ce3 = nn.CrossEntropyLoss(reduction='sum')
        self.relu = nn.ReLU()

    def forward(self, nodes, edge_index, edge_weight=None):
        x = self.conv1(self.embedding.weight, edge_index, edge_weight=edge_weight)
        # x = self.relu(self.conv1(self.embedding.weight, edge_index))
        # x = self.conv2(x, edge_index)
        return x[nodes]

    def loss(self, y_hat, y):
        return self.ce(y_hat, y)

    def losses(self, y_hat, y):
        return self.ce2(y_hat, y)

    def loss_sum(self, y_hat, y):
        return self.ce3(y_hat, y)

    def embeddings(self, nodes=None):
        return self.embedding.weight if nodes is None else self.embedding(nodes)
