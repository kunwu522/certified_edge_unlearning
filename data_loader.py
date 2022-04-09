from copyreg import pickle
import os
import pickle
from collections import defaultdict
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch_geometric.datasets import Planetoid, Coauthor, PolBlogs
from sklearn.model_selection import train_test_split


def initialize_features(dataset, num_nodes, emb_dim):
    features_path = os.path.join('./data', dataset, 'features.pt')
    if os.path.exists(features_path):
        features = torch.load(features_path)
    else:
        features = torch.zeros(num_nodes, emb_dim)
        nn.init.xavier_normal_(features)
        torch.save(features, features_path)
    return features.numpy()


def load_cora(args):
    paper_dict = defaultdict(dict)
    with open(os.path.join('./data/', args.data, 'cora.content'), 'r') as fp:
        for line in fp:
            splitted = line.split()
            paper_id = splitted[0]
            label = splitted[-1]
            feature = list(map(int, splitted[1: -1]))
            paper_dict[paper_id] = {'label': label, 'feature': feature}

    node2idx = {paper_id: idx for idx, paper_id in enumerate(paper_dict.keys())}
    num_nodes = len(node2idx)
    nodes = list(range(num_nodes))

    labels = []
    features = []

    for paper_id, value in paper_dict.items():
        labels.append(value['label'])
        features.append(value['feature'])
    with open(os.path.join('./data', args.data, 'label2idx.dict'), 'rb') as fp:
        label2idx = pickle.load(fp)
    # label2idx = {l: idx for idx, l in enumerate(set(labels))}
    labels = [label2idx[l] for l in labels]
    features = np.array(features)

    edges = []
    with open(os.path.join('./data', args.data, 'cora.cites'), 'r') as fp:
        for line in fp:
            e = line.split()
            edges.append((node2idx[e[0]], node2idx[e[1]]))

    if not args.feature:
        features = initialize_features(args.data, num_nodes, args.emb_dim)
    return nodes, edges, features, labels, node2idx, label2idx


def load_pubmed(args):
    dataset = Planetoid(root='./data', name='PubMed')
    data = dataset[0]

    features = data.x.numpy() if args.feature else initialize_features(args.data, len(data.x), args.emb_dim)

    nodes = list(range(len(data.x)))
    labels = data.y.tolist()
    edges = [(e[0], e[1]) for e in data.edge_index.t().tolist()]

    return nodes, edges, features, labels


def load_physics(args):
    dataset = Coauthor(root='./data/physics', name='Physics')
    data = dataset[0]

    features = data.x.numpy() if args.feature else initialize_features(args.data, len(data.x), args.emb_dim)

    nodes = list(range(len(data.x)))
    labels = data.y.tolist()
    edges = [(e[0], e[1]) for e in data.edge_index.t().tolist()]

    return nodes, edges, features, labels


def load_citeseer(args):
    dataset = Planetoid(root='./data/citeseer', name='Citeseer')
    data = dataset[0]

    features = data.x.numpy() if args.feature else initialize_features(args.data, len(data.x), args.emb_dim)

    nodes = list(range(len(data.x)))
    labels = data.y.tolist()
    edges = [(e[0], e[1]) for e in data.edge_index.t().tolist()]

    return nodes, edges, features, labels


def load_polblogs(args):
    dataset = PolBlogs(root='./data/polblogs')
    data = dataset[0]

    features = initialize_features(args.data, data.num_nodes, args.emb_dim)

    nodes = list(range(data.num_nodes))
    labels = data.y.tolist()
    edges = [(e[0], e[1]) for e in data.edge_index.t().tolist()]

    return nodes, edges, features, labels


def load_data(args):
    if args.data == 'cora':
        nodes, edges, features, labels, _, _ = load_cora(args)
    elif args.data == 'pubmed':
        nodes, edges, features, labels = load_pubmed(args)
    elif args.data == 'physics':
        nodes, edges, features, labels = load_physics(args)
    elif args.data == 'citeseer':
        nodes, edges, features, labels = load_citeseer(args)
    elif args.data == 'polblogs':
        nodes, edges, features, labels = load_polblogs(args)
    else:
        raise ValueError(f'Invalid dataset, {args.data}.')

    train_set_path = os.path.join('./data/', args.data, 'train_set.pt')
    valid_set_path = os.path.join('./data/', args.data, 'valid_set.pt')
    test_set_path = os.path.join('./data/', args.data, 'test_set.pt')
    if os.path.exists(train_set_path) and os.path.exists(valid_set_path) and os.path.exists(test_set_path):
        train_set = torch.load(os.path.join('./data', args.data, 'train_set.pt'))
        valid_set = torch.load(os.path.join('./data', args.data, 'valid_set.pt'))
        test_set = torch.load(os.path.join('./data', args.data, 'test_set.pt'))
    else:
        nodes_train, nodes_test, labels_train, labels_test = train_test_split(nodes, labels, test_size=0.2)
        nodes_train, nodes_valid, labels_train, labels_valid = train_test_split(
            nodes_train, labels_train, test_size=0.2)
        train_set = CoraDataset(nodes_train, labels_train)
        valid_set = CoraDataset(nodes_valid, labels_valid)
        test_set = CoraDataset(nodes_test, labels_test)
        torch.save(train_set, train_set_path)
        torch.save(valid_set, valid_set_path)
        torch.save(test_set, test_set_path)

    data = {
        'nodes': nodes,
        'edges': edges,
        'features': features,
        'labels': labels,
        'num_nodes': len(nodes),
        'num_edges': len(edges),
        'num_classes': np.max(labels) + 1,
        # 'node2idx': node2idx,
        # 'label2idx': label2idx,
        'train_set': train_set,
        'valid_set': valid_set,
        'test_set': test_set,
    }
    return data


class CoraDataset(Dataset):

    def __init__(self, nodes, labels):
        self.nodes = nodes
        self.labels = labels

    def __len__(self):
        return len(self.nodes)

    def __getitem__(self, idx):
        return self.nodes[idx], self.labels[idx]

    def remove(self, node):
        index = self.nodes.index(node)
        del self.nodes[index]
        del self.labels[index]
