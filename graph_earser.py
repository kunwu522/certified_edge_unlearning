'''
    An implementation of Graph Unlearning (https://arxiv.org/abs/2103.14991)

    @ Author: Kun Wu
    @ Date: Mar 7th, 2020

'''
import os
import time
import pickle
import random
import copy
from collections import defaultdict
from tqdm import tqdm
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report
from model import GNN

from train import train_model
from data_loader import CoraDataset
from utils import remove_undirected_edges


def _blpa(nodes, adj_list, k, delta, T):
    ''' Balanced LPA (BLPA)
        Algorithm 1 of paper Graph Unlearning (https://arxiv.org/abs/2103.14991)
        Input:
            nodes: The set of all nodes
            adj_lis: a dictionary that stores the neighbors of each node
            k: number of shards
            delta, maximum number of nodes in each shard
            T: maximum iteration
        Output:
            shards
    '''
    shards = [set() for _ in range(k)]
    node2shard = {}

    for node in nodes:
        random_index = random.choice(range(k))
        shards[random_index].add(node)
        node2shard[node] = random_index

    F = []
    t = 0
    while True:
        for node in nodes:
            for c in shards:
                neighbors = adj_list[node]
                vs = neighbors.intersection(c)
                for v in vs:
                    F.append((node, node2shard[node], node2shard[v], len(vs)))
        sorted_F = sorted(F, key=lambda x: x[3], reverse=True)

        is_shard_changed = False
        for node, index_src, index_dst, _ in sorted_F:
            c_src, c_dst = shards[index_src], shards[index_dst]
            if len(c_dst) < delta:
                c_dst.add(node)
                if node in c_src:
                    c_src.remove(node)
                is_shard_changed = True

        if t > T or not is_shard_changed:
            print('t:', t)
            break
        t += 1

    return [list(s) for s in shards]


def _node_embeddings(args, data, model='gcn', device=torch.device('cpu')):
    embedding_size = args.emb_dim if data['features'] is None else data['features'].shape[1]
    model = GNN(data['num_nodes'], embedding_size,
                args.hidden, 32, data['features'], args.feature_update, model)
    model = model.to(device)
    edge_index = torch.tensor(data['edges'], device=device).t()
    nodes = torch.tensor(data['nodes'], device=device)
    model.eval()
    with torch.no_grad():
        embeddings = model(nodes, edge_index)
    return embeddings.cpu().numpy()


def _bekm(args, data, k, delta, T):
    embeddings = _node_embeddings(args, data)

    nodes = data['nodes']
    centoid_embs = embeddings[random.sample(nodes, k)]

    t = 0
    shards = [set() for _ in range(k)]
    while True:
        F = {}
        for i, emb in zip(nodes, embeddings):
            for j, c_emb in enumerate(centoid_embs):
                F[(i, j)] = np.linalg.norm(emb - c_emb)
        sorted_F = {k: v for k, v in sorted(F.items(), key=lambda x: x[1])}

        for i, j in sorted_F.keys():
            if len(shards[j]) < delta:
                shards[j].add(i)

        update_centoid_embs = []
        for shard in shards:
            update_centoid_embs.append(np.sum(embeddings[list(shard)], axis=0) / len(shard))

        if t > T or np.linalg.norm(np.array(update_centoid_embs) - np.array(centoid_embs)) < 0.001:
            break

        t += 1
    return [list(s) for s in shards]


def _optimal_aggr():
    pass


def _majority_aggr(data, shard_models, shard_edges, device):
    test_loader = DataLoader(data['test_set'], batch_size=1024, shuffle=False)
    y_preds, y_true = [], []
    for nodes, labels in test_loader:
        nodes, labels = nodes.to(device), labels.to(device)

        shard_predictions = []
        for model, edges in zip(shard_models, shard_edges):
            model = model.to(device)
            edge_index = torch.tensor(edges, device=device).t()
            y_hat = model(nodes, edge_index)
            y_pred = torch.argmax(y_hat, dim=1)
            shard_predictions.append(y_pred.unsqueeze(0))

        shard_predictions = torch.cat(shard_predictions, dim=0).t()
        predictions = torch.mode(shard_predictions)[0]

        y_preds.extend(predictions.cpu().tolist())
        y_true.extend(labels.cpu().tolist())

    result = classification_report(y_true, y_preds, digits=4, output_dict=True)
    return result['accuracy'], result['macro avg']['precision'], result['macro avg']['recall'], result['macro avg']['f1-score']


def _mean_aggr(data, shard_models, shard_edges, device):
    test_loader = DataLoader(data['test_set'], batch_size=1024, shuffle=False)
    y_preds, y_true = [], []
    for nodes, labels in test_loader:
        nodes, labels = nodes.to(device), labels.to(device)
        all_y_hat = []
        for model, edges in zip(shard_models, shard_edges):
            model = model.to(device)
            edge_index = torch.tensor(edges, device=device).t()
            y_hat = model(nodes, edge_index)
            all_y_hat.append(y_hat.unsqueeze(1))
        mean_y_hat = torch.mean(torch.cat(all_y_hat, dim=1), dim=1)
        y_pred = torch.argmax(mean_y_hat, dim=1)
        y_preds.extend(y_pred.cpu().tolist())
        y_true.extend(labels.cpu().tolist())

    result = classification_report(y_true, y_preds, digits=4, output_dict=True)
    return result['accuracy'], result['macro avg']['precision'], result['macro avg']['recall'], result['macro avg']['f1-score']


def train(args, data, device, path='./baseline'):

    # shards_path = os.path.join('./baseline', 'grapheraser', f'{args.data}_shards_{args.partition}.list')
    # if os.path.exists(shards_path):
    #     with open(shards_path, 'rb') as fp:
    #         shards = pickle.load(fp)
    # else:
    nodes = data['nodes']
    edges = data['edges']

    args.num_shards = 20
    args.max_t = 10

    if args.partition == 'blpa':
        adj_list = defaultdict(set)
        for v1, v2 in edges:
            adj_list[v1].add(v2)
            adj_list[v2].add(v1)
        shards = _blpa(nodes, adj_list, args.num_shards, 150, args.max_t)
    elif args.partition == 'bekm':
        shards = _bekm(args, data, args.num_shards, 150, args.max_t)
    # with open(shards_path, 'wb') as fp:
    #     pickle.dump(shards, fp)

    shard_edges = []
    for shard in shards:
        _edges = []
        for v1, v2 in data['edges']:
            if v1 in shard and v2 in shard:
                _edges.append((v1, v2))
        shard_edges.append(_edges)

    shard_models_path = os.path.join(path, 'grapheraser',
                                     f'{args.model}_{args.data}_shard_models_feature_{args.partition}.list'
                                     if args.feature else f'{args.model}_{args.data}_shard_models_{args.partition}.list')
    # if os.path.exists(shard_models_path):
    #     with open(shard_models_path, 'rb') as fp:
    #         shard_models = pickle.load(fp)
    # else:
    t0 = time.time()
    shard_models = []
    for shard, _edges in tqdm(zip(shards, shard_edges), total=len(shards)):
        data_ = copy.deepcopy(data)
        data_['edges'] = _edges
        data_['nodes'] = [v for v in shard]
        labels = [data['labels'][v] for v in shard]
        data_['train_set'] = CoraDataset(shard, labels)
        data_['valid_set'] = data['test_set']
        model = train_model(args, data_, eval=False, verbose=False, device=device)
        shard_models.append(model)
    duration = time.time() - t0
    with open(shard_models_path, 'wb') as fp:
        pickle.dump(shard_models, fp)

    if args.aggr == 'majority':
        acc, precision, recall, f1 = _majority_aggr(data, shard_models, shard_edges, device)
    elif args.aggr == 'mean':
        acc, precision, recall, f1 = _mean_aggr(data, shard_models, shard_edges, device)
    return acc, f1, duration
    # print('Result:')
    # print(
    #     f'  Accuracy: {np.mean(acc):.4f}, Precision: {np.mean(precision):.4f}, Recall: {np.mean(recall):.4f}, F1: {np.mean(f1):.4f}.')


def unlearn(args, data, edges_to_forget, device, path='./baseline'):
    shards_path = os.path.join('./baseline', 'grapheraser', f'{args.data}_shards_{args.partition}.list')
    with open(shards_path, 'rb') as fp:
        shards = pickle.load(fp)

    shard_edges = []
    for shard in shards:
        _edges = []
        for v1, v2 in data['edges']:
            if v1 in shard and v2 in shard:
                _edges.append((v1, v2))
        shard_edges.append(_edges)

    shard_models_path = os.path.join(path, 'grapheraser',
                                     f'{args.model}_{args.data}_shard_models_feature_{args.partition}.list'
                                     if args.feature else f'{args.model}_{args.data}_shard_models_{args.partition}.list')
    with open(shard_models_path, 'rb') as fp:
        shard_models = pickle.load(fp)

    t0 = time.time()
    unlearn_shard_models = []
    unlearn_shard_edges = []
    retrain_count = 0
    num_left_edges_list = []
    num_epochs_list = []
    _args = copy.deepcopy(args)
    _args.epochs = 100
    for model, edges in zip(shard_models, shard_edges):
        overlapped = set(edges).intersection(set(edges_to_forget))
        _edges = remove_undirected_edges(list(set(edges)), edges_to_forget)
        if len(_edges) == 0:
            continue
        if len(overlapped) > 0:
            data_ = copy.deepcopy(data)
            data_['edges'] = _edges
            data_['nodes'] = [v for v in shard]
            labels = [data['labels'][v] for v in shard]
            data_['train_set'] = CoraDataset(shard, labels)
            data_['valid_set'] = data['test_set']
            _model = train_model(_args, data_, eval=False, verbose=False, device=device)
            unlearn_shard_models.append(_model)
            unlearn_shard_edges.append(_edges)
            retrain_count += 1

            num_left_edges_list.append(len(edges))
            # num_epochs_list.append(num_epochs)
        else:
            unlearn_shard_models.append(model)
            unlearn_shard_edges.append(edges)

    # print(f'unlearning duration: {(time.time() - t0):.4f}.')
    if args.aggr == 'majority':
        acc, precision, recall, f1 = _majority_aggr(data, unlearn_shard_models, unlearn_shard_edges, device)
    elif args.aggr == 'mean':
        acc, precision, recall, f1 = _mean_aggr(data, unlearn_shard_models, unlearn_shard_edges, device)

    duration = time.time() - t0
    # print(f'{args.partition}, forgetting {len(edges_to_forget)} edges:')
    # print(f'  Avg # of edge in shards is {np.mean(num_left_edges_list)}.')
    # print(f'  Duration: {duration:.2f}.')
    # print(f'  Avg # of epochs: {np.mean(num_epochs_list)}.')

    # print('Result:')
    # print(
    #     f'  Accuracy: {np.mean(acc):.4f}, Precision: {np.mean(precision):.4f}, Recall: {np.mean(recall):.4f}, F1: {np.mean(f1):.4f}.')
    return duration, acc, f1, retrain_count
