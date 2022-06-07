from audioop import bias
import torch
import torch.nn as nn
from torch_geometric.nn.models import MLP
from torch_geometric.nn import GCNConv, GATConv, SAGEConv, GINConv


class GNN(nn.Module):

    def __init__(self, num_nodes, embedding_size, hidden_sizes, num_classes, weights, feature_update, model, dropout=0.5):

        super(GNN, self).__init__()
        self.feature_update = feature_update

        self.embedding = nn.Embedding(num_nodes, embedding_dim=embedding_size)
        self.embedding.weight = nn.Parameter(torch.from_numpy(weights).float(), requires_grad=feature_update)

        def gnn_layer(model, input_size, out_size, dropout):
            if model == 'gcn':
                gnn = GCNConv(input_size, out_size, bias=True)
            elif model == 'gat':
                gnn = GATConv(input_size, out_size, bias=True, dropout=dropout)
            elif model == 'sage':
                gnn = SAGEConv(input_size, out_size, bias=True)
            elif model == 'gin':
                mlp = MLP([input_size, out_size], dropout=0.5, batch_norm=True)
                # mlp = nn.Sequential(
                #     nn.Linear(input_size, int(input_size)),
                #     nn.ReLU(),
                #     nn.Linear(int(input_size), out_size)
                # )
                # mlp = nn.Linear(input_size, out_size)
                gnn = GINConv(mlp)
            else:
                raise NotImplementedError('Unsupposed GNN', model)
            return gnn

        self.gnns = nn.ModuleList()
        output_size = embedding_size
        for hidden_size in hidden_sizes:
            self.gnns.append(gnn_layer(model, output_size, hidden_size, dropout))
            output_size = hidden_size
        self.gnns.append(gnn_layer(model, output_size, num_classes, dropout))

        self.ce = nn.CrossEntropyLoss()
        self.ce2 = nn.CrossEntropyLoss(reduction='none')
        self.ce3 = nn.CrossEntropyLoss(reduction='sum')
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, nodes, edge_index, v=None, delta=None):
        x = self.embedding.weight
        if v is not None and delta is not None:
            x += delta

        for i, gnn in enumerate(self.gnns):
            if i == len(self.gnns) - 1:
                x = gnn(x, edge_index)
            else:
                x = self.relu(gnn(x, edge_index))
                x = self.dropout(x)
        return x[nodes]

    def loss(self, y_hat, y):
        return self.ce(y_hat, y)

    def losses(self, y_hat, y):
        return self.ce2(y_hat, y)

    def loss_sum(self, y_hat, y):
        return self.ce3(y_hat, y)

    def embeddings(self, nodes=None):
        return self.embedding.weight if nodes is None else self.embedding(nodes)

    def reset_parameters(self, weights):
        for gnn in self.gnns:
            gnn.reset_parameters()
        if self.feature_update:
            self.embedding.weight = nn.Parameter(torch.from_numpy(weights).float(), requires_grad=self.feature_update)


# class GCN(nn.Module):

#     def __init__(self, num_nodes, embedding_size, hidden_sizes, num_classes, weights, feature_update):

#         super(GCN, self).__init__()

#         self.embedding = nn.Embedding(num_nodes, embedding_dim=embedding_size)
#         # self.embedding.weight.requires_grad = False
#         self.embedding.weight = nn.Parameter(torch.from_numpy(weights).float(), requires_grad=feature_update)

#         self.gcns = nn.ModuleList()
#         output_size = embedding_size
#         for hidden_size in hidden_sizes:
#             self.gcns.append(GCNConv(output_size, hidden_size, bias=True))
#             output_size = hidden_size
#         self.gcns.append(GCNConv(output_size, num_classes, bias=True))

#         self.ce = nn.CrossEntropyLoss()
#         self.ce2 = nn.CrossEntropyLoss(reduction='none')
#         self.ce3 = nn.CrossEntropyLoss(reduction='sum')
#         self.relu = nn.ReLU()

#     def forward(self, nodes, edge_index):
#         x = self.embedding.weight
#         for i, gcn in enumerate(self.gcns):
#             if i == len(self.gcns) - 1:
#                 x = gcn(x, edge_index)
#             else:
#                 x = self.relu(gcn(x, edge_index))
#         return x[nodes]

#     def loss(self, y_hat, y):
#         return self.ce(y_hat, y)

#     def losses(self, y_hat, y):
#         return self.ce2(y_hat, y)

#     def loss_sum(self, y_hat, y):
#         return self.ce3(y_hat, y)

#     def embeddings(self, nodes=None):
#         return self.embedding.weight if nodes is None else self.embedding(nodes)


# class GAT(nn.Module):

#     def __init__(self, num_nodes, embedding_size, hidden_size, num_classes, weights, feature_update):
#         super(GAT, self).__init__()

#         self.embedding = nn.Embedding(num_nodes, embedding_dim=embedding_size)
#         self.embedding.weight = nn.Parameter(torch.from_numpy(weights).float(), requires_grad=feature_update)

#         self.conv1 = GATConv(embedding_size, num_classes, bias=True)
#         self.ce = nn.CrossEntropyLoss()
#         self.ce2 = nn.CrossEntropyLoss(reduction='none')
#         self.ce3 = nn.CrossEntropyLoss(reduction='sum')

#     def forward(self, nodes, edge_index):
#         x = self.conv1(self.embedding.weight, edge_index)
#         return x[nodes]

#     def loss(self, y_hat, y):
#         return self.ce(y_hat, y)

#     def losses(self, y_hat, y):
#         return self.ce2(y_hat, y)

#     def loss_sum(self, y_hat, y):
#         return self.ce3(y_hat, y)

#     def embeddings(self, nodes=None):
#         return self.embedding.weight if nodes is None else self.embedding(nodes)
