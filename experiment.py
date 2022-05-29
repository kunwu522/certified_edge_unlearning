from collections import defaultdict
import os
import copy
import random
import warnings
from sklearn.manifold import TSNE
import torch
import numpy as np
from scipy.spatial.distance import jensenshannon
import pandas as pd
from tqdm import tqdm
import torch.nn.functional as F
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import seaborn as sns
from adversarial_attack import adv_retrain_unlearn, adv_unlearn, adversaracy_setting
from argument import argument_parser
import graph_earser
from data_loader import load_data
from hessian import hessian
from linkteller import linkteller_attack
from mia import MIA, build_features, construct_mia_data_original, evaluate_mia, generate_mia_features, sample_member, sample_non_member, sample_partial_graph
from train import train_model, test
from retrain import retrain
from unlearn import influence, unlearn
from utils import JSD, remove_undirected_edges, sample_edges


# the best aggr method of GraphEraser for each combination
aggr_method = {
    ('gcn', 'cora', 'blpa'): 'mean',
    ('gcn', 'cora', 'bekm'): 'mean',
    ('gat', 'cora', 'blpa'): 'mean',
    ('gat', 'cora', 'bekm'): 'majority',
    ('sage', 'cora', 'blpa'): 'mean',
    ('sage', 'cora', 'bekm'): 'mean',
    ('gin', 'cora', 'blpa'): 'mean',
    ('gin', 'cora', 'bekm'): 'mean',
    ('gcn', 'citeseer', 'blpa'): 'mean',
    ('gcn', 'citeseer', 'bekm'): 'mean',
    ('gat', 'citeseer', 'blpa'): 'mean',
    ('gat', 'citeseer', 'bekm'): 'mean',
    ('sage', 'citeseer', 'blpa'): 'mean',
    ('sage', 'citeseer', 'bekm'): 'mean',
    ('gin', 'citeseer', 'blpa'): 'mean',
    ('gin', 'citeseer', 'bekm'): 'mean',
    ('gcn', 'polblogs', 'blpa'): 'mean',
    ('gcn', 'polblogs', 'bekm'): 'mean',
    ('gat', 'polblogs', 'blpa'): 'mean',
    ('gat', 'polblogs', 'bekm'): 'mean',
    ('sage', 'polblogs', 'blpa'): 'mean',
    ('sage', 'polblogs', 'bekm'): 'mean',
    ('gin', 'polblogs', 'blpa'): 'mean',
    ('gin', 'polblogs', 'bekm'): 'mean',
    ('gcn', 'physics', 'blpa'): 'mean',
    ('gcn', 'physics', 'bekm'): 'mean',
    ('gat', 'physics', 'blpa'): 'mean',
    ('gat', 'physics', 'bekm'): 'mean',
    ('sage', 'physics', 'blpa'): 'mean',
    ('sage', 'physics', 'bekm'): 'mean',
    ('gin', 'physics', 'blpa'): 'mean',
    ('gin', 'physics', 'bekm'): 'mean',
    ('gcn', 'cs', 'blpa'): 'mean',
    ('gcn', 'cs', 'bekm'): 'mean',
    ('gat', 'cs', 'blpa'): 'mean',
    ('gat', 'cs', 'bekm'): 'mean',
    ('sage', 'cs', 'blpa'): 'mean',
    ('sage', 'cs', 'bekm'): 'mean',
    ('gin', 'cs', 'blpa'): 'mean',
    ('gin', 'cs', 'bekm'): 'mean',
}


def rq0_effectiveness(args, data, device):
    ''' Measure the l2-distance of parameters between retrained models and unlearned models
        The output of this function is orgnized by dataset
    '''
    result = {
        '# edges': [],
        'target model': [],
        'eculiden-distance': [],
        # 'cosine-similarity': [],
        # 'R-l2-norm': [],
        # 'U-l2-norm': [],
        # 'tsne_x': [],
        # 'tsne_y': [],
        # 'setting': [],
    }

    target_models = ['gcn', 'gat', 'sage', 'gin']
    damping = {
        'gcn': 10, 'gat': 100, 'sage': 10, 'gin': 100
    }
    for model in target_models:
        args.model = model
        args.damping = damping[model]

        # train a original model
        original_model = train_model(args, data, eval=False, verbose=False, device=device)
        original_param = np.concatenate([p.detach().cpu().numpy().flatten() for p in original_model.parameters()])

        edges_to_forget = sample_edges(args, data, method=args.method)
        for num_edges in tqdm(args.edges):
            retrain_model, _ = retrain(args, data, edges_to_forget[:num_edges], device=device)
            unlearn_model, _ = unlearn(args, data, original_model, edges_to_forget[:num_edges], device)

            euclidean_dist_re_ul = []
            retrain_param = np.concatenate([p.detach().cpu().numpy().flatten() for p in retrain_model.parameters()])
            unlearn_param = np.concatenate([p.detach().cpu().numpy().flatten() for p in unlearn_model.parameters()])
            # for a, b in zip(retrain_model.parameters(), unlearn_model.parameters()):
            #     a = a.detach().cpu().numpy()
            #     b = b.detach().cpu().numpy()
            #     euclidean_dist_re_ul.append(np.linalg.norm(a - b))
            #     # row_cosine_sim.append(np.mean([1 - cosine(aa, bb) for aa, bb in zip(a, b)]))

            euclidean_dist_re_ul = np.linalg.norm(retrain_param - unlearn_param)
            euclidean_dist_re_or = np.linalg.norm(retrain_param - original_param)
            # for a, b in zip(retrain_model.parameters(), original_model.parameters()):
            #     a = a.detach().cpu().numpy()
            #     b = b.detach().cpu().numpy()
            #     euclidean_dist_re_or.append(np.linalg.norm(a - b))
            # euclidean_dist_re_or = np.sum(euclidean_dist_re_or)

            # row_cosine_sim = np.mean(row_cosine_sim)
            # R_l2_norm = np.sum([np.linalg.norm(p.detach().cpu().numpy()) for p in retrain_model.parameters()])
            # U_l2_norm = np.sum([np.linalg.norm(p.detach().cpu().numpy()) for p in unlearn_model.parameters()])

            result['# edges'].append(num_edges)
            result['# edges'].append(num_edges)
            result['target model'].append(f'{model}_RE_vs_UL')
            result['target model'].append(f'{model}_RE_vs_OR')
            # result['target model'].append(f'UL_{model}')
            result['eculiden-distance'].append(euclidean_dist_re_ul)
            result['eculiden-distance'].append(euclidean_dist_re_or)
            # result['cosine-similarity'].append(row_cosine_sim)
            # result['R-l2-norm'].append(R_l2_norm)
            # result['U-l2-norm'].append(U_l2_norm)

    df = pd.DataFrame(result)
    df.to_csv(os.path.join('./result', f'rq0_{args.data}.csv'))


def _rq1_efficacy_jsd(args, data, device):
    # args.edges = [100, 200, 400, 800, 1000]

    original_model = train_model(args, data, eval=False, verbose=False, device=device)

    nodes = torch.tensor(data['nodes'], device=device)

    result = defaultdict(list)
    for num_edges in tqdm(args.edges, f'{args.data}-{args.model}'):
        jsd_list = []
        for _ in range(10):
            edges_to_forget = sample_edges(args, data, method='random')[:num_edges]
            edge_index_prime = torch.tensor(remove_undirected_edges(data['edges'], edges_to_forget), device=device).t()

            retrain_model, _ = retrain(args, data, edges_to_forget, device)
            unlearn_model, _ = unlearn(args, data, original_model, edges_to_forget, device)

            retrain_model.eval()
            with torch.no_grad():
                retrain_post = retrain_model(nodes, edge_index_prime)

            unlearn_model.eval()
            with torch.no_grad():
                unlearn_post = unlearn_model(nodes, edge_index_prime)

            retrain_post = F.softmax(retrain_post, dim=1).cpu().numpy().astype(np.float64)
            unlearn_post = F.softmax(unlearn_post, dim=1).cpu().numpy().astype(np.float64)

            _jsd = JSD(retrain_post, unlearn_post)
            # print(np.sum(np.isinf(_jsd)))
            # print(retrain_post[np.argwhere(np.isinf(_jsd)).reshape(-1)])
            # print(unlearn_post[np.argwhere(np.isinf(_jsd)).reshape(-1)])

            if np.sum(_jsd < 0) > 0:
                _jsd[_jsd < 0] = 0
            # jsd = np.mean(jsd[~np.isnan(jsd)])
            jsd_list.append(np.mean(_jsd))
        result['# edges'].append(num_edges)
        result['JSD'].append(np.mean(jsd_list))
    df = pd.DataFrame(data=result)
    print(df)
    df.to_csv(os.path.join('./result', f'rq1_efficacy_jsd_{args.data}_{args.model}.csv'))


def _approximation_evaluate(args, data, device):
    args.edges = [200, 400, 600, 800, 1000, 1200, 1400, 1600, 1800, 2000]
    original_model = train_model(args, data, eval=False, verbose=False, device=device)
    test_loader = DataLoader(data['test_set'], batch_size=args.test_batch, shuffle=False)

    retrain_losses, unlearn_losses = [], []
    for num_edges in tqdm(args.edges, f'{args.data}-{args.model}'):
        for _ in range(10):
            edges_to_forget = sample_edges(args, data, method='random')[:num_edges]
            edge_index_prime = torch.tensor(remove_undirected_edges(data['edges'], edges_to_forget), device=device).t()

            retrain_model, _ = retrain(args, data, edges_to_forget, device)
            unlearn_model, _ = unlearn(args, data, original_model, edges_to_forget, device)

            _, retrain_loss = test(retrain_model, test_loader, edge_index_prime, device)
            _, unlearn_loss = test(unlearn_model, test_loader, edge_index_prime, device)
            retrain_losses.append(retrain_loss)
            unlearn_losses.append(unlearn_loss)

    df = pd.DataFrame({
        'retrain_loss': retrain_losses,
        'unlearn_loss': unlearn_losses,
    })
    df.to_csv(os.path.join('./result', f'appr_loss_{args.data}_{args.model}.csv'))


def _rq1_fidelity(args, data, device):
    result = defaultdict(list)
    test_loader = DataLoader(data['test_set'], batch_size=args.test_batch, shuffle=False)
    edge_index = torch.tensor(data['edges'], device=device).t()

    for _ in tqdm(range(10), desc=f'{args.data}-{args.model}'):
        # train a original model
        original_model = train_model(args, data, eval=False, verbose=False, device=device)
        original_res, _ = test(original_model, test_loader, edge_index, device)

        # add original result as 0 edges
        result['# edges'].append(0)
        result['setting'].append('retrain')
        result['accuracy'].append(original_res['accuracy'])
        result['f1'].append(original_res['macro avg']['f1-score'])

        result['# edges'].append(0)
        result['setting'].append('unlearn')
        result['accuracy'].append(original_res['accuracy'])
        result['f1'].append(original_res['macro avg']['f1-score'])

        # Retraining and unlearning
        edges_to_forget = sample_edges(args, data, method='degree')
        for num_edges in args.edges:
            retrain_model, _ = retrain(args, data, edges_to_forget[:num_edges], device, verbose=args.verbose)
            unlearn_model, _ = unlearn(args, data, original_model, edges_to_forget[:num_edges], device)

            _edges = remove_undirected_edges(data['edges'], edges_to_forget[:num_edges])
            edge_index_prime = torch.tensor(_edges, device=device).t()
            retrain_res, _ = test(retrain_model, test_loader, edge_index_prime, device)
            unlearn_res, _ = test(unlearn_model, test_loader, edge_index_prime, device)

            result['# edges'].append(num_edges)
            result['setting'].append('retrain')
            result['accuracy'].append(retrain_res['accuracy'])
            result['f1'].append(retrain_res["macro avg"]["f1-score"])

            result['# edges'].append(num_edges)
            result['setting'].append('unlearn')
            result['accuracy'].append(unlearn_res['accuracy'])
            result['f1'].append(unlearn_res["macro avg"]["f1-score"])

    df = pd.DataFrame(data=result)
    print(df)
    if args.feature:
        df.to_csv(os.path.join('./result', f'rq1_fidelity_{args.data}_{args.model}.csv'))
    else:
        df.to_csv(os.path.join('./result', f'rq1_fidelity_{args.data}_{args.model}_no-feature.csv'))


def rq1_effectiveness_mia(args, data, device):
    num_edges = args.edges[0]

    # prepare data
    test_loader = DataLoader(data['test_set'], batch_size=1024, shuffle=False)
    edge_index = torch.tensor(data['edges'], device=device).t()

    original_model = train_model(args, data, eval=False, verbose=False, device=device)
    orig_res, _ = test(original_model, test_loader, edge_index, device)
    orig_acc = orig_res['accuracy']
    print(f'The accuracy of original model is: {orig_acc:.4f}.')

    edges_to_forget = sample_edges(args, data, method='random')[:num_edges]
    tmp = 0
    for v1, v2 in edges_to_forget:
        if data['labels'][v1] == data['labels'][v2]:
            tmp += 1
    edges_prime = remove_undirected_edges(data['edges'], edges_to_forget)

    retrain_model, _ = retrain(args, data, edges_to_forget, device)
    unlearn_model, _ = unlearn(args, data, original_model, edges_to_forget, device)

    edge_index_prime = torch.tensor(edges_prime, device=device).t()
    retrain_res, _ = test(retrain_model, test_loader, edge_index_prime, device)
    unlearn_res, _ = test(unlearn_model, test_loader, edge_index_prime, device)
    retrain_acc = retrain_res['accuracy']
    unlearn_acc = unlearn_res['accuracy']
    print(
        f'Finished retrain and unlearn, the accuracy are: retrain({retrain_acc:.4f}) and unlearn({unlearn_acc:.4f}).')

    # MIA attak
    partial_graph = sample_partial_graph(edges_prime, edges_to_forget)
    mia_edges, mia_labels = construct_mia_data_original(partial_graph)

    # train MIA on original GNN
    original_model.eval()
    with torch.no_grad():
        y_hat = original_model(data['nodes'], edge_index)
    posterior_o = F.softmax(y_hat, dim=1)

    test_edges = sample_non_member(data['num_nodes'], edges_prime, edges_to_forget, num_edges) + edges_to_forget
    same_class_count = 0
    for edge in test_edges:
        v1, v2 = edge[0], edge[1]
        if data['labels'][v1] == data['labels'][v2]:
            same_class_count += 1
    o_same_class_rate = same_class_count / len(test_edges)
    x_train_o, y_train_o, x_test_o, y_test_o = generate_mia_features(mia_edges, mia_labels,
                                                                     sample_non_member(
                                                                         data['num_nodes'], edges_prime, edges_to_forget, num_edges) + edges_to_forget,
                                                                     [0] * num_edges + [1] * num_edges,
                                                                     posterior_o.cpu().numpy())
    mia = MIA(x_train_o, y_train_o)
    o_mia_acc, o_mia_tp, o_mia_tn = evaluate_mia(mia, x_test_o, y_test_o)

    # test MIA on retrained GNN
    retrain_model.eval()
    with torch.no_grad():
        y_hat = retrain_model(data['nodes'], edge_index_prime)
    posterior_r = F.softmax(y_hat, dim=1)

    # need to remove undirected edges
    _edges = []
    for v1, v2 in edges_prime:
        if (v1, v2) in _edges or (v2, v1) in _edges:
            continue
        _edges.append((v1, v2))
    test_edges = sample_member(_edges, num_edges) + edges_to_forget
    same_class_count = 0
    for edge in test_edges:
        v1, v2 = edge[0], edge[1]
        if data['labels'][v1] == data['labels'][v2]:
            same_class_count += 1
    r_same_class_rate = same_class_count / len(test_edges)
    x_train_r, y_train_r, x_test_r, y_test_r = generate_mia_features(mia_edges, mia_labels,
                                                                     test_edges, [1] * num_edges + [0] * num_edges,
                                                                     posterior_r.cpu().numpy())
    mia = MIA(x_train_r, y_train_r)
    r_mia_acc, r_mia_tp, r_mia_tn = evaluate_mia(mia, x_test_r, y_test_r)

    # test MIA on unlearned GNN
    unlearn_model.eval()
    with torch.no_grad():
        y_hat = unlearn_model(data['nodes'], edge_index_prime)
    posterior_u = F.softmax(y_hat, dim=1)
    x_train_u, y_train_u, x_test_u, y_test_u = generate_mia_features(mia_edges, mia_labels,
                                                                     test_edges, [1] * num_edges + [0] * num_edges,
                                                                     posterior_u.cpu().numpy())
    mia = MIA(x_train_u, y_train_u)
    u_mia_acc, u_mia_tp, u_mia_tn = evaluate_mia(mia, x_test_u, y_test_u)

    df = pd.DataFrame(data={
        'model': ['Original', 'Retrain', 'Unlearn'],
        'Accuracy': [o_mia_acc, r_mia_acc, u_mia_acc],
        'TPR': [o_mia_tp, r_mia_tp, u_mia_tp],
        'TNR': [o_mia_tn, r_mia_tn, u_mia_tn],
        'SCR': [o_same_class_rate, r_same_class_rate, r_same_class_rate]
    })
    print(df)
    return [o_mia_acc, r_mia_acc, u_mia_acc], [o_mia_tp, r_mia_tp, u_mia_tp], [o_mia_tn, r_mia_tn, u_mia_tn], [o_same_class_rate, r_same_class_rate, r_same_class_rate]


def _rq4_efficiency(args, data, device):
    hiddens = {
        'RI': [[16], [24, 12], [24, 16, 8], [24, 16, 16, 8], [24, 16, 16, 16, 8]],
        'NF': [[16], [24, 12], [24, 16, 8], [24, 16, 16, 8], [24, 16, 16, 16, 8]]
    }
    damping = {
        'RI': {
            'gcn': [10, 100, 100, 100, 100],
            'gat': [100, 1000, 1000, 1000, 1000],
            'sage': [100, 1000, 1000, 1000, 1000],
            'gin': [100, 1000, 100, 1000, 1000],
        },
        'NF': {
            'gcn': [10, 100, 100, 100, 100],
            'gat': [100, 1000, 1000, 1000, 1000],
            'sage': [100, 1000, 1000, 1000, 1000],
            'gin': [100, 1000, 1000, 1000, 1000],
        },
    }

    num_edges = 500
    max_num_layers = len(hiddens['RI']) + 1

    result = defaultdict(list)
    for _ in tqdm(range(10), desc=f'RQ2({args.data}, {args.model})'):
        edges_to_forget = sample_edges(args, data, method='random')[:num_edges]
        for i in range(max_num_layers - 1):
            for setting in ['RI', 'NF']:
                args.hidden = hiddens[setting][i]
                args.feature = setting == 'NF'
                args.no_feature_update = setting == 'RI'

                # train a original model
                original_model = train_model(args, data, eval=False, verbose=False, device=device)
                # retrain
                _, retrain_time = retrain(args, data, edges_to_forget, device)
                # unlearn
                args.damping = damping[setting][args.model][i]
                _, unlearn_time = unlearn(args, data, original_model, edges_to_forget, device=device)

                result['# layers'].append(len(args.hidden))
                result['setting'].append('retrain')
                result['type'].append(setting)
                result['running time'].append(retrain_time)

                result['# layers'].append(len(args.hidden))
                result['setting'].append('ERAEDGE')
                result['type'].append(setting)
                result['running time'].append(unlearn_time)

    df = pd.DataFrame(data=result)
    print(df.groupby(['# layers', 'type', 'setting']).mean())
    df.to_csv(os.path.join('./result', f'rq4_efficiency_{args.data}_{args.model}.csv'))


def _gnn_settings(args, data, device):
    hiddens = {
        'RI': [[16], [24, 12], [24, 16, 8], [24, 16, 16, 8], [24, 16, 16, 16, 8]],
        'NF': [[16], [24, 12], [24, 16, 8], [24, 16, 16, 8], [24, 16, 16, 16, 8]]
    }
    damping = {
        'RI': {
            'gcn': [10, 100, 100, 100, 100],
            'gat': [100, 1000, 1000, 1000, 1000],
            'sage': [100, 1000, 1000, 1000, 1000],
            'gin': [100, 1000, 100, 1000, 1000],
        },
        'NF': {
            'gcn': [10, 100, 100, 100, 100],
            'gat': [100, 1000, 1000, 1000, 1000],
            'sage': [100, 1000, 1000, 1000, 1000],
            'gin': [100, 1000, 1000, 1000, 1000],
        },
    }

    num_edges = 500
    max_num_layers = len(hiddens['RI']) + 1
    test_loader = DataLoader(data['test_set'], batch_size=args.test_batch, shuffle=False)
    edge_index = torch.tensor(data['edges'], device=device).t()

    d = defaultdict(list)
    for _ in tqdm(range(10), desc=f'RQ2({args.data}, {args.model})'):
        edges_to_forget = sample_edges(args, data, method=args.method)[:num_edges]
        for i in range(max_num_layers - 1):
            for setting in ['RI', 'NF']:
                args.hidden = hiddens[setting][i]
                args.feature = setting == 'NF'
                args.no_feature_update = setting == 'RI'

                # train a original model
                original_model = train_model(args, data, eval=False, verbose=False, device=device)
                # retrain
                retrain_model, _ = retrain(args, data, edges_to_forget, device)
                # unlearn
                args.damping = damping[setting][args.model][i]
                unlearn_model, _ = unlearn(args, data, original_model, edges_to_forget, device=device)

                # evaluate
                _edges = remove_undirected_edges(data['edges'], edges_to_forget)
                edge_index_prime = torch.tensor(_edges, device=device).t()
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore')
                    res, _ = test(original_model, test_loader, edge_index, device)
                    retrain_res, _ = test(retrain_model, test_loader, edge_index_prime, device)
                    unlearn_res, _ = test(unlearn_model, test_loader, edge_index_prime, device)

                d['# Layer'].append(i+1)
                d['Setting'].append(setting + '-original')
                d['Accuracy'].append(res['accuracy'])

                d['# Layer'].append(i+1)
                d['Setting'].append(setting + '-retrain')
                d['Accuracy'].append(retrain_res['accuracy'])

                d['# Layer'].append(i+1)
                d['Setting'].append(setting + '-ours')
                d['Accuracy'].append(unlearn_res['accuracy'])
                # print(f'Original model: Accuracy({res["accuracy"]:.4f}), F1({res["macro avg"]["f1-score"]:.4f}).')

    df = pd.DataFrame(d)
    df.to_csv(os.path.join('./result', f'rq2_{args.data}_{args.model}_l{max_num_layers}.csv'))


def _rq2_fidelity(args, data, device):
    args.edges = [100, 200, 400, 800, 1000]

    # train a original model
    original_model = train_model(args, data, eval=False, verbose=False, device=device)
    test_loader = DataLoader(data['test_set'], batch_size=args.test_batch, shuffle=False)

    d = defaultdict(list)
    for _ in tqdm(range(10), desc=f'{args.data}-{args.model}'):
        for method in ['random', 'max-degree', 'min-degree']:
            if method == 'max-degree':
                args.max_degree = True
                args.method = 'degree'
            elif method == 'min-degree':
                args.method = 'degree'
                args.max_degree = False
            else:
                args.method = method

            edges_to_forget = sample_edges(args, data, method=args.method)

            for num_edges in args.edges:
                retrain_model, retrain_time = retrain(args, data, edges_to_forget[:num_edges], device)
                unlearn_model, unlearn_time = unlearn(args, data, original_model, edges_to_forget[:num_edges], device)

                _edges = remove_undirected_edges(data['edges'], edges_to_forget[:num_edges])
                edge_index_prime = torch.tensor(_edges, device=device).t()
                retrain_res, _ = test(retrain_model, test_loader, edge_index_prime, device)
                unlearn_res, _ = test(unlearn_model, test_loader, edge_index_prime, device)

                d['# edges'].append(num_edges)
                d['setting'].append(f'{method}-R')
                d['accuracy'].append(retrain_res['accuracy'])
                d['running time'].append(retrain_time)

                d['# edges'].append(num_edges)
                d['setting'].append(f'{method}-U')
                d['accuracy'].append(unlearn_res['accuracy'])
                d['running time'].append(unlearn_time)

    df = pd.DataFrame(d)
    df.to_csv(os.path.join('./result', f'rq2_fidelity_{args.data}_{args.model}.csv'))


def _rq2_efficiency_unlearn(args, data, device):
    args.edges = [100, 200, 400, 800, 1000]
    original_model = train_model(args, data, eval=False, verbose=False, device=device)

    result = defaultdict(list)
    for _ in tqdm(range(10), desc=f'{args.data}-{args.model}'):
        for method in ['random', 'max-degree', 'min-degree', 'saliency']:
            if method == 'max-degree':
                args.max_degree = True
                args.method = 'degree'
            elif method == 'min-degree':
                args.method = 'degree'
                args.max_degree = False
            else:
                args.method = method
            edges_to_forget = sample_edges(args, data, method=args.method)

            for num_edges in args.edges:
                # _, retrain_time = retrain(args, data, edges_to_forget[:num_edges], device)
                _, unlearn_time = unlearn(args, data, original_model, edges_to_forget[:num_edges], device)
                result['running time'].append(unlearn_time)
                result['# edges'].append(num_edges)
                result['method'].append(method)
    df = pd.DataFrame(data=result)
    print(df.groupby(['# edges', 'method']).mean())
    df.to_csv(os.path.join('./result', 'rq2_efficiency_unlearn.csv'))


def _rq2_efficiency_retrain(args, data, device):
    args.edges = [100, 200, 400, 800, 1000]

    result = defaultdict(list)
    for _ in tqdm(range(10), desc=f'{args.data}-{args.model}'):
        for method in ['random', 'max-degree', 'min-degree', 'saliency']:
            if method == 'max-degree':
                args.max_degree = True
                args.method = 'degree'
            elif method == 'min-degree':
                args.method = 'degree'
                args.max_degree = False
            else:
                args.method = method
            edges_to_forget = sample_edges(args, data, method=args.method)

            for num_edges in args.edges:
                _, retrain_time = retrain(args, data, edges_to_forget[:num_edges], device)
                result['running time'].append(retrain_time)
                result['# edges'].append(num_edges)
                result['method'].append(method)
    df = pd.DataFrame(data=result)
    print(df.groupby(['# edges', 'method']).mean())
    df.to_csv(os.path.join('./result', 'rq2_efficiency_retrain.csv'))


def _rq2_efficacy_mia(args, data, device):
    # train a original model
    original_model = train_model(args, data, eval=False, verbose=False, device=device)

    mia_result = defaultdict(list)
    for _ in tqdm(range(10), desc=f'{args.data}-{args.model}'):
        for method in ['random', 'max-degree', 'min-degree', 'saliency']:
            if method == 'max-degree':
                args.max_degree = True
                args.method = 'degree'
            elif method == 'min-degree':
                args.method = 'degree'
                args.max_degree = False
            else:
                args.method = method

            edges_to_forget = sample_edges(args, data, method=args.method)

            for num_edges in args.edges:
                retrain_model, retrain_time = retrain(args, data, edges_to_forget[:num_edges], device)
                unlearn_model, unlearn_time = unlearn(args, data, original_model, edges_to_forget[:num_edges], device)

                o_pl, r_pl, u_pl = _mia_attack(data, original_model, retrain_model,
                                               unlearn_model, edges_to_forget[:num_edges], device)
                mia_result['# edges'].append(num_edges)
                mia_result['setting'].append(f'{method}-O')
                mia_result['privacy leakage'].append(o_pl)

                mia_result['# edges'].append(num_edges)
                mia_result['setting'].append(f'{method}-R')
                mia_result['privacy leakage'].append(r_pl)

                mia_result['# edges'].append(num_edges)
                mia_result['setting'].append(f'{method}-U')
                mia_result['privacy leakage'].append(u_pl)

    mia_df = pd.DataFrame(mia_result)
    print(mia_df.groupby(['# edges', 'setting']).mean())
    mia_df.to_csv(os.path.join('./result', f'rq2_mia_{args.data}_{args.model}.csv'))


def _rq2_efficacy_jsd(args, data, device):
    args.edges = [100, 200, 300, 400, 500]

    # train a original model
    original_model = train_model(args, data, eval=False, verbose=False, device=device)
    nodes = torch.tensor(data['nodes'], device=device)

    jsd_result = defaultdict(list)
    for _ in tqdm(range(10), desc=f'{args.data}-{args.model}'):
        for method in ['random', 'max-degree', 'min-degree', 'saliency']:
            if method == 'max-degree':
                args.max_degree = True
                args.method = 'degree'
            elif method == 'min-degree':
                args.method = 'degree'
                args.max_degree = False
            else:
                args.method = method

            edges_to_forget = sample_edges(args, data, method=args.method)

            for num_edges in args.edges:
                retrain_model, retrain_time = retrain(args, data, edges_to_forget[:num_edges], device)
                unlearn_model, unlearn_time = unlearn(args, data, original_model, edges_to_forget[:num_edges], device)

                _edges = remove_undirected_edges(data['edges'], edges_to_forget[:num_edges])
                edge_index_prime = torch.tensor(_edges, device=device).t()
                retrain_model.eval()
                with torch.no_grad():
                    retrain_post = retrain_model(nodes, edge_index_prime)
                unlearn_model.eval()
                with torch.no_grad():
                    unlearn_post = unlearn_model(nodes, edge_index_prime)
                retrain_post = F.softmax(retrain_post, dim=1).cpu().numpy()
                unlearn_post = F.softmax(unlearn_post, dim=1).cpu().numpy()

                _jsd = JSD(retrain_post, unlearn_post)
                if np.sum(_jsd < 0) > 0:
                    _jsd[_jsd < 0] = 0
                jsd_result['# edges'].append(num_edges)
                jsd_result['jsd'].append(np.mean(_jsd))

    jsd_df = pd.DataFrame(jsd_result)
    jsd_df.to_csv(os.path.join('./result', f'rq2_jsd_{args.data}_{args.model}.csv'))


def edge_type_analysis(args):
    for d in args.datasets:
        args.data = d
        data = load_data(args)

        directed_edges = []
        for v1, v2 in data['edges']:
            if (v1, v2) in directed_edges or (v2, v1) in directed_edges:
                continue
            directed_edges.append((v1, v2))
        num_same_class = 0
        for v1, v2 in directed_edges:
            if data['labels'][v1] == data['labels'][v2]:
                num_same_class += 1
        print(f'In {d}, the same-class rate is: {(num_same_class / len(directed_edges)):.4f}.')


def analyze_utility():
    for target in ['gcn', 'gat', 'sage', 'gin']:
        for data in ['cora', 'citeseer', 'polblogs']:
            df = pd.read_csv(os.path.join('./result', f'rq1_unlearn_{data}_{target}_l1_16.csv'))
            retrain_acc = df['retrain-acc'].values
            unlearn_acc = df['unlearn-acc'].values
            avg_acc_dff = np.mean(np.abs(retrain_acc - unlearn_acc))
            print(f'{target} on {data} of Avg. Accuracy Diff:', avg_acc_dff)
    print('------------------------------------------------------------------')

    # print('---------------------------BLPA VS Unlearn (Acc)--------------------------------')
    # for target in ['gcn', 'gat', 'sage']:
    #     for data in ['cora', 'polblogs']:
    #         df = pd.read_csv(os.path.join('./result', f'rq1_{data}_{target}.csv'))
    #         unlearn_acc = df['unlearn-acc'].values
    #         blpa_acc = df['blpa-acc'].values
    #         avg_acc_dff = np.mean(np.abs(unlearn_acc - blpa_acc))
    #         print(f'{target} on {data} of Avg. Accuracy Diff:', avg_acc_dff / np.mean(blpa_acc))


def analyze_running_time():
    # for target in ['gcn', 'gat', 'sage', 'gin']:
    #     for data in ['cora', 'citeseer', 'polblogs']:
    #         df = pd.read_csv(os.path.join('./result', f'rq1_unlearn_{data}_{target}_l1_16.csv'))
    #         retrain_time = df['retrain-time'].values
    #         unlearn_time = df['unlearn-time'].values
    #         avg_time_dff = np.mean(np.abs(retrain_time - unlearn_time))
    #         print(f'{target} on {data} of Avg. time diff:', avg_time_dff / np.mean(retrain_time))

    times = []
    for target in ['gcn', 'gat', 'sage', 'gin']:
        for data in ['cora', 'citeseer', 'polblogs']:
            df = pd.read_csv(os.path.join('./result', f'rq1_fidelity_baseline_{data}_{target}.csv'))
            df = df[df['# edges'].isin([100, 200, 400, 800, 1000])]
            times.extend(df['running time'])
    print('Max', np.max(times))
    print('Min', np.min(times))


def _rq3_efficacy(args, data, device):
    args.edges = [100, 200, 400, 800, 1000]
    test_loader = DataLoader(data['train_set'], shuffle=False, batch_size=args.test_batch)
    edge_index = torch.tensor(data['edges'], device=device).t()
    nodes = torch.tensor(data['nodes'], device=device)

    jsd_result = defaultdict(list)
    mia_result = defaultdict(list)
    for _ in tqdm(range(10), desc=f'{args.data}-{args.model}'):
        original_model = train_model(args, data, eval=False, verbose=False, device=device)
        original_res, _ = test(original_model, test_loader, edge_index, device=device)
        for num_edges in args.edges:
            def _efficacy(retrain_model, unlearn_model, edge_index_prime):
                retrain_model.eval()
                with torch.no_grad():
                    retrain_post = retrain_model(nodes, edge_index_prime)
                unlearn_model.eval()
                with torch.no_grad():
                    unlearn_post = unlearn_model(nodes, edge_index_prime)

                retrain_post = F.softmax(retrain_post, dim=1).cpu().numpy()
                unlearn_post = F.softmax(unlearn_post, dim=1).cpu().numpy()

                _jsd = JSD(retrain_post, unlearn_post)
                if np.sum(_jsd < 0) > 0:
                    _jsd[_jsd < 0] = 0
                return np.mean(_jsd)

            adv_model, adv_unlearn_model, A = adv_unlearn(args, data, num_edges, device)
            adv_o_pl, adv_r_pl, adv_u_pl = _mia_attack(data, adv_model, original_model,
                                                       adv_unlearn_model, A, device)
            edge_index_prime = torch.tensor(remove_undirected_edges(data['edges'], A), device=device).t()

            adv_jsd = _efficacy(original_model, adv_unlearn_model, edge_index_prime)

            # Benign
            random_edges = []
            while len(random_edges) < num_edges * 2:
                v1 = random.randint(0, data['num_nodes'] - 1)
                v2 = random.randint(0, data['num_nodes'] - 1)
                if (v1, v2) in data['edges'] or (v2, v1) in data['edges']:
                    continue
                random_edges.append((v1, v2))
                random_edges.append((v2, v1))

            _data = copy.deepcopy(data)
            _data['edges'] += random_edges
            benign_orig = train_model(args, _data, eval=False, verbose=False, device=device)
            benign_unlearn, _ = unlearn(args, _data, benign_orig, random_edges, device=device)

            _edges = remove_undirected_edges(data['edges'], random_edges)
            edge_index_prime = torch.tensor(_edges, device=device).t()
            beni_o_pl, beni_r_pl, beni_u_pl = _mia_attack(data, benign_orig, original_model,
                                                          benign_unlearn, random_edges, device)
            benign_jsd = _efficacy(original_model, benign_unlearn, edge_index_prime)

            jsd_result['# edges'].append(num_edges)
            jsd_result['setting'].append('adv')
            jsd_result['jsd'].append(np.mean(adv_jsd))

            jsd_result['# edges'].append(num_edges)
            jsd_result['setting'].append('benign')
            jsd_result['jsd'].append(np.mean(benign_jsd))

            mia_result['# edges'].append(num_edges)
            mia_result['setting'].append('adv')
            mia_result['privacy leakage'].append(adv_u_pl)

            mia_result['# edges'].append(num_edges)
            mia_result['setting'].append('benign')
            mia_result['privacy leakage'].append(beni_u_pl)

    jsd_df = pd.DataFrame(jsd_result)
    jsd_df.to_csv(os.path.join('./result', f'rq3_jsd_{args.data}_{args.model}.csv'))

    mia_df = pd.DataFrame(mia_result)
    mia_df.to_csv(os.path.join('./result', f'rq3_mia_{args.data}_{args.model}.csv'))


def _rq3_degree(args, data, device):
    args.edges = [200]
    test_loader = DataLoader(data['train_set'], shuffle=False, batch_size=args.test_batch)
    edge_index = torch.tensor(data['edges'], device=device).t()

    result = defaultdict(list)
    for _ in tqdm(range(10), desc=f'{args.data}-{args.model}'):
        original_model = train_model(args, data, eval=False, verbose=False, device=device)
        original_res, _ = test(original_model, test_loader, edge_index, device=device)
        for num_edges in args.edges:
            # Adv
            adv_model, adv_unlearn_model, A = adv_unlearn(args, data, num_edges, device)

            # Benign
            random_edges = []
            while len(random_edges) < num_edges * 2:
                v1 = random.randint(0, data['num_nodes'] - 1)
                v2 = random.randint(0, data['num_nodes'] - 1)
                if (v1, v2) in data['edges'] or (v2, v1) in data['edges']:
                    continue
                random_edges.append((v1, v2))
                random_edges.append((v2, v1))

            node_degree = defaultdict(int)
            for edge in data['edges']:
                node_degree[edge[0]] += 1
                node_degree[edge[1]] += 1

            adv_degree = [(node_degree[v1] + node_degree[v2]) / 2 for v1, v2 in A]
            benign_degree = [(node_degree[v1] + node_degree[v2]) / 2 for v1, v2 in random_edges]

            result['setting'].append('adv')
            result['min degree'].append(np.min(adv_degree))
            result['max degree'].append(np.max(adv_degree))
            result['avg degree'].append(np.mean(adv_degree))

            result['setting'].append('benign')
            result['min degree'].append(np.min(benign_degree))
            result['max degree'].append(np.max(benign_degree))
            result['avg degree'].append(np.mean(benign_degree))

    df = pd.DataFrame(result)
    print(df.groupby('setting').mean())


def _rq3_fidelity(args, data, device):
    args.edges = [100, 200, 400, 800, 1000]
    test_loader = DataLoader(data['test_set'], shuffle=False, batch_size=args.test_batch)
    edge_index = torch.tensor(data['edges'], device=device).t()

    result = defaultdict(list)
    for _ in tqdm(range(10), desc=f'{args.data}-{args.model}'):
        original_model = train_model(args, data, eval=False, verbose=False, device=device)
        original_res, _ = test(original_model, test_loader, edge_index, device=device)
        for num_edges in args.edges:
            _, _, adv_res, _ = adv_retrain_unlearn(args, data, num_edges, device)

            # Benign
            random_edges = []
            while len(random_edges) < num_edges * 2:
                v1 = random.randint(0, data['num_nodes'] - 1)
                v2 = random.randint(0, data['num_nodes'] - 1)
                if (v1, v2) in data['edges'] or (v2, v1) in data['edges']:
                    continue
                random_edges.append((v1, v2))
                random_edges.append((v2, v1))

            _data = copy.deepcopy(data)
            _data['edges'] += random_edges
            benign_orig = train_model(args, _data, eval=False, verbose=False, device=device)
            benign_unlearn, _ = unlearn(args, _data, benign_orig, random_edges, device=device)

            benign_res, _ = test(benign_unlearn, test_loader, edge_index, device)
            result['# edges'].append(num_edges)
            result['setting'].append('Retrain')
            result['accuracy'].append(original_res['accuracy'])

            result['# edges'].append(num_edges)
            result['setting'].append('Benign')
            result['accuracy'].append(benign_res['accuracy'])

            result['# edges'].append(num_edges)
            result['setting'].append('Advesarial')
            result['accuracy'].append(adv_res['accuracy'])

    df = pd.DataFrame(data=result)
    print(df.groupby(['# edges', 'setting']).mean())
    df.to_csv(f'./result/rq3_fidelity_{args.data}_{args.model}.csv')


def _adversarial_setting(args, data, device):
    args.edges = [100, 200, 400, 800, 1000]
    adv_origianl_acc, adv_retrain_acc, adv_unlearn_acc = adversaracy_setting(args, data, device)
    df = pd.DataFrame({
        '# edges': args.edges + args.edges + args.edges,
        'setting': ['Adversarial'] * len(args.edges) + ['Original Model'] * len(args.edges) + ['Unlearn'] * len(args.edges),
        'Accuracy': adv_origianl_acc + adv_retrain_acc + adv_unlearn_acc,
    })

    if args.hidden:
        df.to_csv(os.path.join(
            './result', f'rq4_unlearn_{args.data}_{args.model}_h{len(args.hidden)}_{"_".join(map(str, args.hidden))}.csv'))
    else:
        df.to_csv(os.path.join('./result', f'rq4_unlearn_{args.data}_{args.model}.csv'))

    # edges_to_forget = sample_edges(args, data, method='random')
    unlearn_acc = []
    for num_edges in tqdm(args.edges, desc='retrain'):
        random_edges = []
        while len(random_edges) < num_edges * 2:
            v1 = random.randint(0, data['num_nodes'] - 1)
            v2 = random.randint(0, data['num_nodes'] - 1)
            if (v1, v2) in data['edges'] or (v2, v1) in data['edges']:
                continue
            random_edges.append((v1, v2))
            random_edges.append((v2, v1))

        _data = copy.deepcopy(data)
        _data['edges'] += random_edges
        bengin_orig = train_model(args, _data, eval=False, verbose=False, device=device)
        bengin_unlearn, _ = unlearn(args, _data, bengin_orig, random_edges, device=device)

        test_loader = DataLoader(data['test_set'], batch_size=args.test_batch, shuffle=False)
        edge_index = torch.tensor(data['edges'], device=device).t()
        res, loss = test(bengin_unlearn, test_loader, edge_index, device)
        unlearn_acc.append(res['accuracy'])

    # bengin_diff = np.abs(np.array(adv_retrain_acc) - np.array(unlearn_acc))
    # adv_diff = np.abs(np.array(adv_retrain_acc) - np.array(adv_unlearn_acc))

    df = pd.DataFrame({
        '# edges': args.edges + args.edges + args.edges,
        'type': ['Retrain'] * len(args.edges) + ['Benign'] * len(args.edges) + ['Adversarial'] * len(args.edges),
        'Model Accuracy': np.concatenate((adv_retrain_acc, unlearn_acc, adv_unlearn_acc))
    })

    if args.hidden:
        df.to_csv(os.path.join(
            './result', f'rq4_diff_{args.data}_{args.model}_h{len(args.hidden)}_{"_".join(map(str, args.hidden))}.csv'))
    else:
        df.to_csv(os.path.join('./result', f'rq4_diff_{args.data}_{args.model}.csv'))


def cosine_similarity_mat(A, B):
    cs = []
    for a, b in zip(A, B):
        cs.append(cosine_similarity(a, b))
    return np.mean(cs)


def cosine_similarity(a, b):
    p1 = np.sqrt(np.sum(a ** 2))
    p2 = np.sqrt(np.sum(b ** 2))
    return np.dot(a, b) / (p1 * p2)


def _baseline(args, data, device):

    result = defaultdict(list)
    for _ in tqdm(range(10), desc=f'{args.data}-{args.model}'):
        for partition in ['blpa', 'bekm']:
            args.partition = partition
            args.aggr = aggr_method[(args.model, args.data, partition)]
            acc, f1, base_time = graph_earser.train(args, data, device)
            # print(f'{partition} original: {acc:.4f}, {f1:.4f}')
            result['# edges'].append(0)
            result['partition'].append(partition)
            result['running time'].append(base_time)
            result['accuracy'].append(acc)
            result['f1'].append(f1)
            result['# shards'].append(20)

        edges_to_forget = sample_edges(args, data, method='random')
        # 2. retrain
        for num_edges in tqdm(args.edges, desc='baseline'):
            for partition in ['blpa', 'bekm']:
                args.partition = partition
                args.aggr = aggr_method[(args.model, args.data, partition)]
                base_time, base_acc, base_f1, num_shards = graph_earser.unlearn(
                    args, data, edges_to_forget[:num_edges], device)
                result['# edges'].append(num_edges)
                result['partition'].append(partition)
                result['running time'].append(base_time)
                result['accuracy'].append(base_acc)
                result['f1'].append(base_f1)
                result['# shards'].append(num_shards)

    baseline_df = pd.DataFrame(data=result)
    print('Baseline Result:')
    print(baseline_df)
    if args.feature:
        baseline_df.to_csv(os.path.join('./result', f'rq1_fidelity_baseline_{args.data}_{args.model}.csv'))
    else:
        baseline_df.to_csv(os.path.join('./result', f'rq1_fidelity_baseline_{args.data}_{args.model}_no-feature.csv'))


def _baseline_comparison(args, data, device):

    original_model = train_model(args, data, eval=False, verbose=args.verbose, device=device)

    # baseline: GraphEraser, original
    for partition in ['blpa', 'bekm']:
        args.partition = partition
        args.aggr = aggr_method[(args.model, args.data, partition)]
        acc, f1 = graph_earser.train(args, data, device, './result')
        print(f'{partition} original: {acc:.4f}, {f1:.4f}')

    result = defaultdict(list)
    for _ in tqdm(range(100), desc=f'{args.data}-{args.model}'):
        edges_to_forget = sample_edges(args, data)[:100]

        retrain_model, retrain_time = retrain(args, data, edges_to_forget, device=device, verbose=args.verbose)
        result['setting'].append('Retrain')
        result['running time'].append(retrain_time)

        unlearn_model, unlearn_time = unlearn(args, data, original_model, edges_to_forget, device=device)
        result['setting'].append('ERAEDGE')
        result['running time'].append(unlearn_time)

        for partition in ['blpa', 'bekm']:
            args.partition = partition
            args.aggr = aggr_method[(args.model, args.data, partition)]
            base_time, base_acc, base_f1, num_shards = graph_earser.unlearn(
                args, data, edges_to_forget, device, path='./result')
            result['setting'].append(partition.upper())
            result['running time'].append(base_time)

    df = pd.DataFrame(data=result)
    df.to_csv(os.path.join('./result', f'comparison_{args.data}_{args.model}.csv'))


def _mia_attack(data, original_model, retrain_model, unlearn_model, edges_to_forget, device):
    nodes = torch.tensor(data['nodes'], device=device)
    edge_index = torch.tensor(data['edges'], device=device).t()
    edges_prime = remove_undirected_edges(data['edges'], edges_to_forget)
    edge_index_prime = torch.tensor(edges_prime, device=device).t()

    # sample edges
    pos_edges = random.sample(edges_prime, len(edges_to_forget))
    neg_edges = []
    while len(neg_edges) < len(edges_to_forget):
        v1 = random.choice(data['nodes'])
        v2 = random.choice(data['nodes'])
        if (v1, v2) in data['edges'] or (v2, v1) in data['edges']:
            continue
        neg_edges.append((v1, v2))

    o_acc, o_tpr, o_tnr = linkteller_attack('original', nodes, data['features'], edge_index, original_model,
                                            edges_to_forget, neg_edges, device=device)
    r_acc, r_tpr, r_tnr = linkteller_attack('retrain', nodes, data['features'], edge_index_prime, retrain_model,
                                            pos_edges, edges_to_forget, device=device)
    u_acc, u_tpr, u_tnr = linkteller_attack('unlearn', nodes, data['features'], edge_index_prime, unlearn_model,
                                            pos_edges, edges_to_forget, device=device)
    return o_tpr, 1 - r_tnr, 1 - u_tnr


def _rq1_efficacy_mia(args, data, device):
    args.edges = [100, 200, 400]
    origin_model = train_model(args, data, eval=False, verbose=False, device=device)
    result = defaultdict(list)
    for _ in tqdm(range(10), desc=f'{args.data}-{args.model}'):
        edges_to_forget = sample_edges(args, data, method='random')
        for num_edges in args.edges:
            retrain_model, _ = retrain(args, data, edges_to_forget[:num_edges], device)
            unlearn_model, _ = unlearn(args, data, origin_model, edges_to_forget[:num_edges], device)

            # Get privacy leakage (pl)
            origin_pl, retrain_pl, unlearn_pl = _mia_attack(
                data, origin_model, retrain_model, unlearn_model, edges_to_forget[:num_edges], device)

            result['# edges'].append(num_edges)
            result['setting'].append('Original')
            result['privacy leakage'].append(origin_pl)

            result['# edges'].append(num_edges)
            result['setting'].append('Retrain')
            result['privacy leakage'].append(retrain_pl)

            result['# edges'].append(num_edges)
            result['setting'].append('Unlearn')
            result['privacy leakage'].append(unlearn_pl)

    df = pd.DataFrame(result)
    print(df.groupby(['# edges', 'setting']).mean())
    df.to_csv(os.path.join('./result', f'rq1_efficacy_mia_{args.data}_{args.model}.csv'))


def _evaluate_unlearn_time(args, data, device):
    args.edges = [100, 200, 400, 800, 1000]

    result = defaultdict(list)
    for _ in tqdm(range(10), desc=f'unlearn-{args.data}-{args.model}'):
        edges_to_forget = sample_edges(args, data, method='random')
        original_model = train_model(args, data, eval=False, verbose=False, device=device)

        for num_edges in args.edges:
            _, unlearn_time = unlearn(args, data, original_model, edges_to_forget[:num_edges], device)
            result['# edges'].append(num_edges)
            result['running time'].append(unlearn_time)
        df = pd.DataFrame(data=result)

    print(df.groupby(['# edges']).mean())
    df.to_csv(os.path.join('./result', f'rq1_efficiency_unlearn_{args.data}_{args.model}_l1.csv'))


def _evaluate_retraining_time(args, data, device):
    args.edges = [100, 200, 400, 800, 1000]
    result = defaultdict(list)
    for _ in tqdm(range(10), desc=f'{args.data}-{args.model}'):
        edges_to_forget = sample_edges(args, data, method='random')

        for num_edges in args.edges:
            _, retrain_time, num_epochs = retrain(
                args, data, edges_to_forget[:num_edges], device=device, return_epoch=True)
            result['# edges'].append(num_edges)
            result['running time'].append(retrain_time)
            result['# epochs'].append(num_epochs)
        df = pd.DataFrame(data=result)

    print(df.groupby(['# edges']).mean())
    df.to_csv(os.path.join('./result', f'rq1_efficiency_retrain_{args.data}_{args.model}_l1.csv'))


def _evaluate_original_mdoel(args, data, device):
    test_loader = DataLoader(data['test_set'], shuffle=False, batch_size=args.test_batch)
    edge_index = torch.tensor(data['edges'], device=device).t()

    result = defaultdict(list)
    for _ in tqdm(range(10), desc=f'retrain-{args.data}-{args.model}'):
        model = train_model(args, data, eval=False, verbose=False, device=device)
        res, _ = test(model, test_loader, edge_index, device)
        result['# edges'].append(0)
        result['accuracy'].append(res['accuracy'])

    df = pd.DataFrame(data=result)
    df.to_csv(os.path.join('./result', f'original_{args.data}_{args.model}.csv'))


def analyze_edges(args):
    args.model = 'gcn'
    result = defaultdict(list)
    for d in tqdm(['cora', 'citeseer', 'polblogs']):
        args.data = d
        data = load_data(args)

        for num_edges in [100, 200, 400, 800, 1000]:
            for method in ['random', 'max-degree', 'min-degree', 'saliency']:
                if method.startswith('max'):
                    args.method = 'degree'
                    args.max_degree = True
                elif method.startswith('min'):
                    args.method = 'degree'
                    args.max_degree = False
                else:
                    args.method = method
                edges_to_forget = sample_edges(args, data, method=args.method)[:num_edges]

                original_node_degree = defaultdict(int)
                for edge in data['edges']:
                    original_node_degree[edge[0]] += 1
                    original_node_degree[edge[1]] += 1

                _edges = remove_undirected_edges(data['edges'], edges_to_forget)
                _nodes = set()
                for v1, v2 in _edges:
                    _nodes.add(v1)
                    _nodes.add(v2)

                node_degree = defaultdict(int)
                for edge in _edges:
                    node_degree[edge[0]] += 1
                    node_degree[edge[1]] += 1

                original_avg_degree = [original_node_degree[n] for n in _nodes]
                avg_degree = [node_degree[n] for n in _nodes]

                result['data'].append(d)
                result['# edges'].append(num_edges)
                result['method'].append(method)
                result['original'].append(np.mean(original_avg_degree))
                result['removed'].append(np.mean(avg_degree))

    df = pd.DataFrame(result)
    print(df)


def analyze_retrain_output(args, data, device):
    result = defaultdict(list)
    for _ in tqdm(range(10), desc=f'{args.data}-{args.model}'):
        nodes = torch.tensor(data['test_set'].nodes, device=device)

        edges_to_forget = sample_edges(args, data, method='random')
        prediction = {}
        for num_edges in args.edges:
            retrain_model, _ = retrain(args, data, edges_to_forget[:num_edges], device)

            edge_prime = remove_undirected_edges(data['edges'], edges_to_forget[:num_edges])
            edge_index_prime = torch.tensor(edge_prime, device=device).t()

            retrain_model.eval()
            with torch.no_grad():
                o = retrain_model(nodes, edge_index_prime)
                y_pred = torch.argmax(o, dim=1)
                prediction[num_edges] = y_pred.cpu().numpy()

        for i in range(1, len(args.edges)):
            e1 = args.edges[i-1]
            e2 = args.edges[i]
            num_changed = np.sum(prediction[e1] != prediction[e2])

            result['# edges'].append(f'{e1} to {e2}')
            result['target'].append(args.model)
            result['dataset'].append(args.data)
            result['# changed'].append(num_changed)
    df = pd.DataFrame(result)
    print(df.groupby(['# edges', 'dataset', 'target']).mean())


def analyze_influence(args, data, device):
    result = defaultdict(list)
    for _ in tqdm(range(2), desc=f'{args.data}-{args.model}'):
        original_model = train_model(args, data, eval=False, verbose=False, device=device)
        edges_to_forget = sample_edges(args, data, method='degree')
        for num_edges in args.edges:
            retrain_model, _ = retrain(args, data, edges_to_forget[:num_edges], device)
            _, _, influence = unlearn(args, data, original_model, edges_to_forget[:num_edges], device, influence=True)
            influence_truth = [torch.linalg.norm((o.detach() - r.detach())).cpu()
                               for o, r in zip(original_model.parameters(), retrain_model.parameters())]
            for i, (inf, t) in enumerate(zip(influence, influence_truth)):
                result['# edges'].append(num_edges)
                result['param'].append(i)
                result['influence'].append(inf.item())
                result['truth'].append(t.item())

    df = pd.DataFrame(result)
    print(df.groupby(['# edges', 'param']).mean())


def analyze_unlearn_output(args, data, device):
    result = defaultdict(list)
    for _ in tqdm(range(10), desc=f'{args.data}-{args.model}'):
        original_model = train_model(args, data, eval=False, verbose=False, device=device)
        nodes = torch.tensor(data['test_set'].nodes, device=device)

        edges_to_forget = sample_edges(args, data, method='random')
        prediction = {}
        for num_edges in args.edges:
            unlearn_model, _ = unlearn(args, data, original_model, edges_to_forget[:num_edges], device=device)

            edge_prime = remove_undirected_edges(data['edges'], edges_to_forget[:num_edges])
            edge_index_prime = torch.tensor(edge_prime, device=device).t()

            unlearn_model.eval()
            with torch.no_grad():
                o = unlearn_model(nodes, edge_index_prime)
                y_pred = torch.argmax(o, dim=1)
                prediction[num_edges] = y_pred.cpu().numpy()

        for i in range(1, len(args.edges)):
            e1 = args.edges[i-1]
            e2 = args.edges[i]
            num_changed = np.sum(prediction[e1] != prediction[e2])

            result['# edges'].append(f'{e1} to {e2}')
            result['target'].append(args.model)
            result['dataset'].append(args.data)
            result['# changed'].append(num_changed)
    df = pd.DataFrame(result)
    print(df.groupby(['# edges', 'dataset', 'target']).mean())


def analyze_hessian(args, data, device):
    model = train_model(args, data, eval=False, verbose=False, device=device)

    x_train = torch.tensor(data['train_set'].nodes, device=device)
    y_train = torch.tensor(data['train_set'].labels, device=device)

    edges_to_forget = sample_edges(args, data, method='degree')
    prediction = {}
    for num_edges in args.edges:
        edge_prime = remove_undirected_edges(data['edges'], edges_to_forget[:num_edges])
        edge_index_prime = torch.tensor(edge_prime, device=device).t()
        # y_hat = model(x_train, edge_index_prime)
        H, _ = hessian(model, edge_index_prime, x_train, y_train, [p for p in model.parameters()])
        for idx, h in enumerate(H):
            print('  H:', h.size())
            if h.dim() == 4:
                n = h.size(0) * h.size(1)
            elif h.dim() == 2:
                n = h.size(0)
            elif h.dim() == 6:
                n = h.size(0) * h.size(1) * h.size(2)
            if h.size(0) == 2708:
                # print('condition number:', np.linalg.cond(h.view(n, -1).cpu().numpy()))
                continue
            E = torch.linalg.eigvals(h.view(n, -1)).real
            print('E', E.tolist())
            print('condition number:', (torch.max(E) / torch.min(E)).item())
            print('max eigenvalue:', torch.max(E).item())
            print('min eigenvalue:', torch.min(E).item())


damping = {
    'cora': {'gcn': 0, 'gat': 0, 'sage': 0, 'gin': 0},
    'citeseer': {'gcn': 0, 'gat': 0, 'sage': 0, 'gin': 0},
    'polblogs': {'gcn': 0, 'gat': 0, 'sage': 0, 'gin': 0},
    'cs': {'gcn': 0, 'gat': 0, 'sage': 0, 'gin': 0},
    'physics': {'gcn': 0, 'gat': 0, 'sage': 0, 'gin': 0},
}


no_feature_damping = {
    'cora': {'gcn': 0, 'gat': 0.01, 'sage': 0, 'gin': 0},
    'citeseer': {'gcn': 0, 'gat': 0.01, 'sage': 0, 'gin': 0},
    'polblogs': {'gcn': 0, 'gat': 0.01, 'sage': 0, 'gin': 0},
    'cs': {'gcn': 0, 'gat': 0.01, 'sage': 0, 'gin': 0},
    'physics': {'gcn': 0, 'gat': 0.01, 'sage': 0, 'gin': 0},
}

if __name__ == '__main__':
    parser = argument_parser()
    parser.add_argument('-rq', type=str, default=None)
    parser.add_argument('-analysis', type=str, default=None)
    parser.add_argument('-mia', dest='mia_attack', action='store_true',
                        help='Indicator of evaluting the unlearning model via MIA attack (accuracy).')
    parser.add_argument('-cosine', dest='cosine', action='store_true')
    parser.add_argument('-datasets', type=str, nargs='+', default=['cora', 'citeseer'])
    parser.add_argument('-targets', type=str, nargs='+', default=['gcn', 'sage', 'gin'])
    # parser.add_argument('-no-baseline', dest='baseline', action='store_false')

    args = parser.parse_args()
    print('Parameters:', vars(args))

    # random.seed(args.seed)
    # np.random.seed(args.seed)
    # torch.manual_seed(args.seed)

    device = torch.device(f'cuda:{args.gpu}') if torch.cuda.is_available() and args.gpu >= 0 else torch.device('cpu')

    for d in args.datasets:
        args.data = d
        data = load_data(args)
        for target in args.targets:
            args.model = target
            if args.feature:
                args.damping = damping[d][target]
            else:
                args.damping = no_feature_damping[d][target]

            if args.rq == 'rq1_fidelity':
                _rq1_fidelity(args, data, device)
            if args.rq == 'rq1_baseline':
                _baseline(args, data, device)

            if args.rq == 'original':
                _evaluate_original_mdoel(args, data, device)
            if args.rq == 'rq1_efficiency':
                _evaluate_retraining_time(args, data, device)
                # _evaluate_unlearn_time(args, data, device)
            if args.rq == 'rq1_mia':
                _rq1_efficacy_mia(args, data, device)
            if args.rq == 'rq1_jsd':
                _rq1_efficacy_jsd(args, data, device)

            if args.rq == 'rq2_efficiency':
                # _rq2_efficiency_retrain(args, data, device)
                _rq2_efficiency_unlearn(args, data, device)

            if args.rq == 'rq2_fidelity':
                _rq2_fidelity(args, data, device)

            if args.rq == 'rq2_mia':
                _rq2_efficacy_mia(args, data, device)

            if args.rq == 'rq2_jsd':
                _rq2_efficacy_jsd(args, data, device)

            if args.rq == 'rq3_fidelity':
                _rq3_fidelity(args, data, device)

            if args.rq == 'rq3_efficacy':
                _rq3_efficacy(args, data, device)

            if args.rq == 'rq3_degree':
                _rq3_degree(args, data, device)

            if args.rq == 'rq4_efficiency':
                _rq4_efficiency(args, data, device)

            elif args.rq == 'gnn_setting':
                _gnn_settings(args, data, device)
            elif args.rq == 'adv':
                _adversarial_setting(args, data, device)
            elif args.rq == 'loss':
                _approximation_evaluate(args, data, device)
            elif args.rq == 'comparison':
                _baseline_comparison(args, data, device)

    for d in args.datasets:
        args.data = d
        data = load_data(args)
        for target in args.targets:
            args.model = target
            if args.feature:
                args.damping = damping[d][target]
            else:
                args.damping = no_feature_damping[d][target]

            if args.analysis == 'time':
                analyze_running_time()
            if args.analysis == 'utility':
                analyze_utility()
            if args.analysis == 'edge_types':
                analyze_edges(args)
            if args.analysis == 'output':
                analyze_unlearn_output(args, data, device)
                # analyze_retrain_output(args, data, device)
            if args.analysis == 'influence':
                analyze_influence(args, data, device)

            if args.analysis == 'hessian':
                analyze_hessian(args, data, device)
    # if args.analysis == 'original':
    #     analyze_original()

    if args.mia_attack:
        data = load_data(args)
        result = defaultdict(list)
        for target in args.targets:
            args.damping = damping[target]
            args.model = target
            for _ in range(10):
                # o_scr, r_scr, u_scr = rq1_effectiveness_mia(args, data, device)
                # result[f'{target}_o_scr'].append(o_scr)
                # result[f'{target}_r_scr'].append(r_scr)
                # result[f'{target}_u_scr'].append(u_scr)
                acc, tpr, tnr, scr = rq1_effectiveness_mia(args, data, device)
                result['model'].extend([target] * 3)
                result['setting'].extend(['original', 'retrain', 'unlearn'])
                result['Accuracy'].extend(acc)
                result['TPR'].extend(tpr)
                result['TNR'].extend(tnr)
                result['SCR'].extend(scr)

        df = pd.DataFrame(result)
        print(df)
        df.to_csv(os.path.join('./result', f'mia_{args.data}.csv'))
