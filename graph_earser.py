'''
    An implementation of Graph Unlearning (https://arxiv.org/abs/2103.14991)

    @ Author: Kun Wu
    @ Date: Mar 7th, 2020

'''
import argparse
import os
import copy
import sys
import time
import math
import logging
import random
import pickle
from collections import defaultdict
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import numpy as np
import torch
import torch.nn.functional as F
import networkx as nx
from torch_geometric.data import Data
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report
from model import GNN
from train import train_model
from data_loader import load_data, CoraDataset

from lib.exp import Exp
from lib.lib_utils import utils
from lib.lib_utils.utils import connected_component_subgraphs
from lib.lib_graph_partition.constrained_lpa_base import ConstrainedLPABase
from lib.lib_aggregator.aggregator import Aggregator
from utils import remove_undirected_edges, sample_edges


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
            # print('t:', t)
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

    shards_path = os.path.join('./baseline', 'grapheraser', f'{args.data}_shards_{args.partition}.list')
    # if os.path.exists(shards_path):
    #     with open(shards_path, 'rb') as fp:
    #         shards = pickle.load(fp)
    # else:
    nodes = data['nodes']
    edges = data['edges']

    args.num_shards = 20 if args.data in ['cora', 'citeseer'] else 50
    args.max_t = 10

    node_threshold = math.ceil(data['num_nodes'] / args.num_shards + 0.005 * (
        data['num_nodes'] - data['num_nodes'] / args.num_shards))

    if args.partition == 'blpa':
        adj_list = defaultdict(set)
        for v1, v2 in edges:
            adj_list[v1].add(v2)
            adj_list[v2].add(v1)
        shards = _blpa(nodes, adj_list, args.num_shards, node_threshold, args.max_t)
    elif args.partition == 'bekm':
        shards = _bekm(args, data, args.num_shards, node_threshold, args.max_t)
    with open(shards_path, 'wb') as fp:
        pickle.dump(shards, fp)

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
    _args = copy.deepcopy(args)
    _args.epochs = 100
    # _args.early_stop = False
    _args.lr = 0.001
    # _args.l2 = 1E-4

    random_indices = list(range(len(shards)))
    random.shuffle(random_indices)
    num_samples = 1
    t0 = time.time()
    for i in random_indices[:num_samples]:
        shard, _edges = shards[i], shard_edges[i]
        data_ = copy.deepcopy(data)
        data_['edges'] = _edges
        data_['nodes'] = [v for v in shard]
        labels = [data['labels'][v] for v in shard]
        x_train, x_valid, y_train, y_valid = train_test_split(shard, labels, test_size=0.1)
        data_['train_set'] = CoraDataset(x_train, y_train)
        data_['valid_set'] = CoraDataset(x_valid, y_valid)
        _ = train_model(_args, data_, eval=False, verbose=False, device=device)
    duration = time.time() - t0
    return duration, num_samples

    # t0 = time.time()
    # shard_models = []
    # for shard, _edges in tqdm(zip(shards, shard_edges), total=len(shards)):
    #     data_ = copy.deepcopy(data)
    #     data_['edges'] = _edges
    #     data_['nodes'] = [v for v in shard]
    #     labels = [data['labels'][v] for v in shard]
    #     x_train, x_valid, y_train, y_valid = train_test_split(shard, labels, test_size=0.1)
    #     data_['train_set'] = CoraDataset(x_train, y_train)
    #     data_['valid_set'] = CoraDataset(x_valid, y_valid)
    #     model = train_model(_args, data_, eval=False, verbose=True, device=device)
    #     shard_models.append(model)
    # duration = time.time() - t0
    # with open(shard_models_path, 'wb') as fp:
    #     pickle.dump(shard_models, fp)

    # if args.aggr == 'majority':
    #     acc, precision, recall, f1 = _majority_aggr(data, shard_models, shard_edges, device)
    # elif args.aggr == 'mean':
    #     acc, precision, recall, f1 = _mean_aggr(data, shard_models, shard_edges, device)
    # return acc, f1, duration
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
    _args.early_stop = False
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


logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        # logging.FileHandler("debug.log"),
        logging.StreamHandler(sys.stdout)
    ]
)


class ExpNodeEdgeUnlearning(Exp):
    def __init__(self, args, data, device):
        ''' This implmenetaion is from paper, Graph Unlearning.
            https://github.com/MinChen00/Graph-Unlearning

            In this experiment, we adopt their edge unlearning as our baselines by applying some modifications.

        '''
        super(ExpNodeEdgeUnlearning, self).__init__(args)
        # self.logger = logging.getLogger('exp_node_edge_unlearning')
        # self.logger.setLevel(logging.INFO)

        self.target_model_name = self.args.model
        self.data = data
        self.device = device

        self.load_data()
        self.determine_target_model()
        # self.run_exp()

    def run_exp(self):
        # unlearning efficiency
        # run_f1 = np.empty((0))
        original_acc = np.empty((0))
        unlearn_acc = np.empty((0))
        unlearning_time = np.empty((0))
        for run in range(args.num_runs):
            logging.info("Run %f" % run)
            self.train_target_models(run)
            aggregate_acc_score = self.aggregate(run)
            original_acc = np.append(original_acc, aggregate_acc_score)

            edge_unlearning_time = self.retrain_affect_target_model(run)
            print('unlearning time:', edge_unlearning_time)
            aggregate_unlearn_acc = self.aggregate(run)
            unlearn_acc = np.append(unlearn_acc, aggregate_unlearn_acc)

            # run_f1 = np.append(run_f1, aggregate_f1_score)
            unlearning_time = np.append(unlearning_time, np.average(edge_unlearning_time))
        # model utility
        # self.f1_score_avg = np.average(run_f1)
        # self.f1_score_std = np.std(run_f1)
        # self.acc_score_avg = np.average(run_acc)
        # self.acc_score_std = np.std(run_acc)
        # self.unlearning_time_avg = np.average(unlearning_time)
        # self.unlearning_time_std = np.std(unlearning_time)
        # logging.info(
        #     "%s %s %s %s" % (self.f1_score_avg, self.f1_score_std, self.unlearning_time_avg, self.unlearning_time_std))
        return original_acc, unlearn_acc, unlearning_time

    def load_data(self):
        self.shard_data = self.data_store.load_shard_data()
        # self.raw_data = self.data_store.load_raw_data()
        # self.train_data = self.data_store.load_train_data()
        self.unlearned_shard_data = self.shard_data

    def determine_target_model(self):
        # num_feats = self.train_data.num_features
        # num_classes = len(self.train_data.y.unique())
        embedding_size = self.args.emb_dim if self.data['features'] is None else self.data['features'].shape[1]
        self.target_model = GNN(self.data['num_nodes'], embedding_size,
                                self.args.hidden, self.data['num_classes'], self.data['features'], self.args.feature_update, self.target_model_name)
        # if not self.args['is_use_batch']:
        #     if self.target_model_name == 'SAGE':
        #         self.target_model = SAGE(num_feats, num_classes)
        #     elif self.target_model_name == 'GCN':
        #         self.target_model = GCN(num_feats, num_classes)
        #     elif self.target_model_name == 'GAT':
        #         self.target_model = GAT(num_feats, num_classes)
        #     elif self.target_model_name == 'GIN':
        #         self.target_model = GIN(num_feats, num_classes)
        #     else:
        #         raise Exception('unsupported target model')
        # else:
        #     if self.target_model_name == 'MLP':
        #         self.target_model = MLP(num_feats, num_classes)
        #     else:
        #         self.target_model = NodeClassifier(num_feats, num_classes, self.args)

    def train_target_models(self, run):
        # if self.args['is_train_target_model']:
        if True:
            logging.info('training target models')

            self.time = {}
            for shard in range(self.args.num_shards):
                self.time[shard] = self._train_model(run, shard)

    def retrain_affect_target_model(self, run):
        if self.args.is_train_target_model and self.args.num_shards != 1:
            self.community_to_node = self.data_store.load_community_data()

            # To fairly compare with baselines, we use our method to sample edges
            unlearned_edges = sample_edges(self.args, self.data, method=self.args.method)[:self.args.num_deleted_edges]

            belong_community = []
            for v1, v2 in unlearned_edges:
                for community, node in self.community_to_node.items():
                    if np.in1d(v1, node).any() and np.in1d(v2, node):
                        belong_community.append(community)

            # calculate the total unlearning time and group unlearning time
            edge_unlearning_time = []
            for shard in range(self.args.num_shards):
                if belong_community.count(shard) != 0:
                    _time = self._train_model(run, shard)
                    edge_unlearning_time.append(_time)

            return edge_unlearning_time

    def aggregate(self, run):
        logging.info('aggregating submodels')

        # posteriors, true_label = self.generate_posterior()
        aggregator = Aggregator(run, self.target_model, self.data, self.unlearned_shard_data, self.args, self.device)
        aggregator.generate_posterior()
        self.aggregate_acc_score = aggregator.aggregate()

        logging.info("Final Test F1: %s" % (self.aggregate_acc_score,))
        return self.aggregate_acc_score

    def _generate_unlearning_request(self, num_unlearned="assign"):
        node_list = []
        for key, value in self.community_to_node.items():
            # node_list.extend(value.tolist())
            node_list.extend(value)

        if num_unlearned == "assign":
            # num_of_unlearned_nodes = self.args['num_unlearned_nodes']
            num_of_unlearned_edges = self.args.num_deleted_edges
        elif num_unlearned == "ratio":
            num_of_unlearned_nodes = int(self.args['ratio_unlearned_nodes'] * len(node_list))

        if self.args['unlearning_request'] == 'random':
            unlearned_nodes_indices = np.random.choice(node_list, num_of_unlearned_nodes, replace=False)

        elif self.args['unlearning_request'] == 'top1':
            sorted_shards = sorted(self.community_to_node.items(), key=lambda x: len(x[1]), reverse=True)
            unlearned_nodes_indices = np.random.choice(sorted_shards[0][1], num_of_unlearned_nodes, replace=False)

        elif self.args['unlearning_request'] == 'adaptive':
            sorted_shards = sorted(self.community_to_node.items(), key=lambda x: len(x[1]), reverse=True)
            candidate_list = np.concatenate([sorted_shards[i][1]
                                            for i in range(int(self.args['num_shards']/2)+1)], axis=0)
            unlearned_nodes_indices = np.random.choice(candidate_list, num_of_unlearned_nodes, replace=False)

        elif self.args['unlearning_request'] == 'last5':
            sorted_shards = sorted(self.community_to_node.items(), key=lambda x: len(x[1]), reverse=False)
            candidate_list = np.concatenate([sorted_shards[i][1]
                                            for i in range(int(self.args['num_shards']/2)+1)], axis=0)
            unlearned_nodes_indices = np.random.choice(candidate_list, num_of_unlearned_nodes, replace=False)

        return unlearned_nodes_indices

    def _unlearning_time_statistic(self):
        ''' Deprecated.
            We aims to unlearn edges.

        '''
        if self.args.is_train_target_model and self.args.num_shards != 1:
            self.community_to_node = self.data_store.load_community_data()
            # random sample 5% nodes, find their belonging communities
            unlearned_nodes = self._generate_unlearning_request(num_unlearned="assgin")
            belong_community = []
            for sample_node in range(len(unlearned_nodes)):
                for community, node in self.community_to_node.items():
                    if np.in1d(unlearned_nodes[sample_node], node).any():
                        belong_community.append(community)

            # calculate the total unlearning time and group unlearning time
            group_unlearning_time = []
            node_unlearning_time = []
            for shard in range(self.args['num_shards']):
                if belong_community.count(shard) != 0:
                    group_unlearning_time.append(self.time[shard])
                    node_unlearning_time.extend([float(self.time[shard]) for j in range(belong_community.count(shard))])
            return node_unlearning_time

        elif self.args.is_train_target_model and self.args.num_shards == 1:
            return self.time[0]

        else:
            return 0

    def _train_model(self, run, shard, unlearned_edges=None):
        logging.info('training target models, run %s, shard %s' % (run, shard))

        start_time = time.time()
        data = self.unlearned_shard_data[shard]
        if unlearned_edges is not None:
            data = copy.deepcopy(self.unlearned_shard_data[shard])
            edges_prime = remove_undirected_edges(data.edge_index.T.tolist(), unlearned_edges)
            data.edge_index = torch.tensor(edges_prime, device=device).t()
        else:
            data = self.unlearned_shard_data[shard]
        self._train(data)
        train_time = time.time() - start_time

        self.data_store.save_target_model(run, self.target_model, shard)

        return train_time

    def _train(self, data, num_epoch=100):
        self.target_model.train()
        self.target_model.reset_parameters(self.data['features'])
        self.target_model = self.target_model.to(self.device)
        x, edge_index, edge_weight = data.x, data.edge_index, data.edge_attr
        x = torch.tensor(x, device=device)
        edge_index = edge_index.to(self.device)
        y = torch.from_numpy(data.y).to(self.device)

        print('x:', x.size(), torch.sum(data.train_mask))

        optimizer = torch.optim.Adam(self.target_model.parameters(), lr=0.01)
        for epoch in range(num_epoch):
            logging.info('epoch %s' % (epoch,))

            optimizer.zero_grad()
            output = self.target_model(x, edge_index)[data.train_mask]
            loss = F.nll_loss(output, y[data.train_mask])
            loss.backward()
            optimizer.step()

            train_acc, test_acc = self._evaluate(data)
            logging.warning('train acc: %s, test acc: %s' % (train_acc, test_acc))

    def _evaluate(self, data):
        self.target_model.eval()
        self.target_model = self.target_model.to(self.device)

        x, edge_index, edge_weight = data.x, data.edge_index, data.edge_attr
        x = torch.tensor(x, device=device)
        edge_index = edge_index.to(self.device)
        y = torch.from_numpy(data.y).to(self.device)

        logits, accs = self.target_model(x, edge_index), []

        for _, mask in data('train_mask', 'test_mask'):
            pred = logits[mask].max(1)[1]
            acc = pred.eq(y[mask]).sum().item() / mask.sum().item()
            accs.append(acc)

        return accs


class ExpGraphPartition(Exp):
    def __init__(self, args, data, device):
        super(ExpGraphPartition, self).__init__(args)

        # self.logger = logging.getLogger('exp_graph_partition')

        # self.load_data()
        self.data = data
        self.train_indices = np.array(self.data['train_set'].nodes)
        self.test_indices = np.array(self.data['test_set'].nodes)
        self.device = device
        # self.train_test_split()
        self.gen_train_graph()
        self.graph_partition()
        self.generate_shard_data()

    def load_data(self):
        self.data = self.data_store.load_raw_data()

    def _train_test_split(self):
        if self.args['is_split']:
            logging.info('splitting train/test data')
            self.train_indices, self.test_indices = train_test_split(
                np.arange((self.data.num_nodes)), test_size=self.args['test_ratio'], random_state=100)
            self.data_store.save_train_test_split(self.train_indices, self.test_indices)

            self.data.train_mask = torch.from_numpy(np.isin(np.arange(self.data.num_nodes), self.train_indices))
            self.data.test_mask = torch.from_numpy(np.isin(np.arange(self.data.num_nodes), self.test_indices))
        else:
            self.train_indices, self.test_indices = self.data_store.load_train_test_split()

            self.data.train_mask = torch.from_numpy(np.isin(np.arange(self.data.num_nodes), self.train_indices))
            self.data.test_mask = torch.from_numpy(np.isin(np.arange(self.data.num_nodes), self.test_indices))

    def gen_train_graph(self):
        # delete ratio of edges and update the train graph
        if self.args.num_deleted_edges != 0:
            logging.debug("Before edge deletion. train data  #.Nodes: %f, #.Edges: %f" % (
                self.data['num_nodes'], self.data['num_edges']))

            # self._ratio_delete_edges()
            # self.data.edge_index = self._ratio_delete_edges(self.data.edge_index)
            edges_ul = sample_edges(self.args, self.data, method=self.args.method)[:self.args.num_deleted_edges]
            self.edges_prime = remove_undirected_edges(self.data['edges'], edges_ul)
            self.edge_index = torch.tensor(self.edges_prime, device=self.device).t()
        else:
            self.edge_index = torch.tensor(self.data['edges'], device=self.device).t()

        # decouple train test edges.
        edge_index = self.edge_index.cpu().numpy()
        test_edge_indices = np.logical_or(np.isin(edge_index[0], self.test_indices),
                                          np.isin(edge_index[1], self.test_indices))

        print('test edges:', np.sum(test_edge_indices))
        train_edge_indices = np.logical_not(test_edge_indices)
        print('train edges:', np.sum(train_edge_indices))
        edge_index_train = edge_index[:, train_edge_indices]

        self.train_graph = nx.Graph()
        self.train_graph.add_nodes_from(self.train_indices)

        # use largest connected graph as train graph
        # if self.args['is_prune']:
        #     self._prune_train_set()

        # reconstruct a networkx train graph
        for u, v in np.transpose(edge_index_train):
            self.train_graph.add_edge(u, v)

        logging.debug("After edge deletion. train graph  #.Nodes: %f, #.Edges: %f" % (
            self.train_graph.number_of_nodes(), self.train_graph.number_of_edges()))
        logging.debug("After edge deletion. train data  #.Nodes: %f, #.Edges: %f" % (
            self.data['num_nodes'], self.data['num_edges']))
        # self.data_store.save_train_data(self.data)
        self.data_store.save_train_graph(self.train_graph)

    def graph_partition(self):
        if self.args.is_partition:
            logging.info('graph partitioning')

            start_time = time.time()
            partition = GraphPartition(self.args, self.train_graph, self.data)
            self.community_to_node = partition.graph_partition()
            partition_time = time.time() - start_time
            logging.info("Partition cost %s seconds." % partition_time)
            self.data_store.save_community_data(self.community_to_node)
        else:
            self.community_to_node = self.data_store.load_community_data()

    def generate_shard_data(self):
        logging.info('generating shard data')

        self.shard_data = {}
        for shard in range(self.args.num_shards):
            train_shard_indices = list(self.community_to_node[shard])
            shard_indices = np.union1d(train_shard_indices, self.test_indices)

            x = np.array(self.data['nodes'])[shard_indices]
            y = np.array(self.data['labels'])[shard_indices]
            edge_index = utils.filter_edge_index_1(self.data, shard_indices)

            data = Data(x=x, edge_index=torch.from_numpy(edge_index), y=y)
            data.train_mask = torch.from_numpy(np.isin(shard_indices, train_shard_indices))
            data.test_mask = torch.from_numpy(np.isin(shard_indices, self.test_indices))

            self.shard_data[shard] = data

        self.data_store.save_shard_data(self.shard_data)

    def _prune_train_set(self):
        # extract the the maximum connected component
        logging.debug("Before Prune...  #. of Nodes: %f, #. of Edges: %f" % (
            self.train_graph.number_of_nodes(), self.train_graph.number_of_edges()))

        self.train_graph = max(connected_component_subgraphs(self.train_graph), key=len)

        logging.debug("After Prune... #. of Nodes: %f, #. of Edges: %f" % (
            self.train_graph.number_of_nodes(), self.train_graph.number_of_edges()))
        # self.train_indices = np.array(self.train_graph.nodes)

    # def _ratio_delete_edges(self, edge_index):
    #     edge_index = edge_index.numpy()

    #     unique_indices = np.where(edge_index[0] < edge_index[1])[0]
    #     unique_indices_not = np.where(edge_index[0] > edge_index[1])[0]
    #     remain_indices = np.random.choice(unique_indices,
    #                                       int(unique_indices.shape[0] * (1.0 - self.args['ratio_deleted_edges'])),
    #                                       replace=False)

    #     remain_encode = edge_index[0, remain_indices] * edge_index.shape[1] * 2 + edge_index[1, remain_indices]
    #     unique_encode_not = edge_index[1, unique_indices_not] * \
    #         edge_index.shape[1] * 2 + edge_index[0, unique_indices_not]
    #     sort_indices = np.argsort(unique_encode_not)
    #     remain_indices_not = unique_indices_not[sort_indices[np.searchsorted(
    #         unique_encode_not, remain_encode, sorter=sort_indices)]]
    #     remain_indices = np.union1d(remain_indices, remain_indices_not)

    #     # self.data.edge_index = torch.from_numpy(edge_index[:, remain_indices])
    #     return torch.from_numpy(edge_index[:, remain_indices])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-exp', type=str, default='None')
    parser.add_argument('-g', dest='gpu', type=int, default=-1)
    parser.add_argument('-d', dest='data', type=str, default='cora')
    parser.add_argument('-m', dest='model', type=str, default='gcn')

    # For training
    parser.add_argument('-hidden', type=int, nargs='+', default=[])
    parser.add_argument('--num_epochs', type=int, default=100)
    parser.add_argument('--num_runs', type=int, default=1)
    # parser.add_argument('-batch', type=int, default=512)
    # parser.add_argument('-test-batch', type=int, default=1024)
    # parser.add_argument('-lr', type=float, default=0.001)
    # parser.add_argument('-l2', type=float, default=1E-5)
    parser.add_argument('-emb-dim', type=int, default=32)
    parser.add_argument('-feature', dest='feature', action='store_true')
    parser.add_argument('-no-feature-update', dest='feature_update', action='store_false')
    # parser.add_argument('-p', dest='patience', type=int, default=20)
    parser.add_argument('--no_train_target_model', dest='is_train_target_model', action='store_false')

    # GraphEraser
    parser.add_argument('-edges', type=int, nargs='+', default=[100, 200, 400, 800, 1000],
                        help='in terms of precentage, how many edges to sample.')
    parser.add_argument('-p', dest='partition_method', type=str, default='lpa_base')
    parser.add_argument('-no-partition', dest='is_partition', action='store_false')
    # parser.add_argument('-k', dest='num_shards', type=int, default=20)
    parser.add_argument('-t', dest='max_t', type=int, default=10)
    parser.add_argument('-method', dest='method', type=str, default='degree')
    parser.add_argument('-max-degree', action='store_true')
    parser.add_argument('--shard_size_delta', type=float, default=0.005)
    parser.add_argument('--num_unlearned_nodes', type=int, default=1)
    parser.add_argument('--ratio_unlearned_nodes', type=float, default=0.005)
    parser.add_argument('--num_unlearned_edges', type=int, default=1)
    parser.add_argument('--ratio_deleted_edges', type=float, default=0.9)
    parser.add_argument('--num_opt_samples', type=int, default=1000)
    parser.add_argument('--terminate_delta', type=int, default=0)
    parser.add_argument('--no-constrained', dest='is_constrained', action='store_false')
    parser.add_argument('--aggregator', type=str, default='optimal', choices=['mean', 'majority', 'optimal'])
    parser.add_argument('--opt_lr', type=float, default=0.001)
    parser.add_argument('--opt_decay', type=float, default=0.0001)
    parser.add_argument('--opt_num_epochs', type=int, default=50)
    parser.add_argument('--num_shards', type=int, default=20)

    args = parser.parse_args()
    print('Arguments:', vars(args))

    data = load_data(args)
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() and args.gpu >= 0 else 'cpu')

    args.num_deleted_edges = 0
    ExpGraphPartition(args, data, device)
    for num_edges in tqdm(args.edges):
        args.num_deleted_edges = num_edges
        exp = ExpNodeEdgeUnlearning(args, data, device)
        original_acc, unlearn_acc, unlearning_time = exp.run_exp()
        print('orignial acc', original_acc)
        print('unlearn acc', unlearn_acc)
        print('unlearn time:', unlearning_time)
