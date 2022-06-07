from collections import defaultdict
import os
import time
import random
import warnings
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch.nn.functional as F
from torch.utils.data import DataLoader
from adversarial_attack import adversaracy_setting
from argument import argument_parser
import graph_earser
from data_loader import load_data
from mia import MIA, evaluate_mia, generate_mia_features, sample_member, sample_non_member
from train import train_model, test
from retrain import loss_difference_node, retrain, loss_difference, retrain_node, epsilons
from unlearn import batch_unlearn, influences_node, node_unlearn, unlearn, influences
from utils import create_model, edges_remove_nodes, find_loss_difference_node, model_path, remove_undirected_edges, save_model, sample_edges, find_loss_difference, sample_nodes


if __name__ == '__main__':
    parser = argument_parser()
    parser.add_argument('-node-unlearn', action='store_true')
    parser.add_argument('-nodes', type=int, nargs='+', default=[50, 100, 150])
    args = parser.parse_args()
    print('Parameters:', vars(args))

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    data = load_data(args)

    if args.data_info:
        print('***************************************')
        print(f' dataset: {args.data}')
        print(f'   # nodes: {data["num_nodes"]}')
        print(f'   # edges: {data["num_edges"]}')
        print(f'   # classes: {data["num_classes"]}')
        print('***************************************')
        print()

    if args.model_info:
        model = create_model(args, data)
        print('***************************************')
        print(f' dataset: {args.model}')
        print(f'   # parameters: {[p.size() for p in model.parameters()]}')
        print(f'   # hidden size: {args.hidden}')
        print('***************************************')
        print()

    device = torch.device(f'cuda:{args.gpu}') if torch.cuda.is_available() and args.gpu >= 0 else torch.device('cpu')

    if args.train:
        # args.early_stop = False
        # args.epochs = 100
        model = train_model(args, data, device=device)
        if args.save:
            save_model(args, model, 'original')

    if args.retrain:
        edges_to_retrain = sample_edges(args, data, method=args.method, num_edges=args.edges)
        retrain(args, data, edges_to_retrain, device)
        loss_difference(args, data, edges_to_retrain, device)
        epsilons(args, data, edges_to_retrain, device)

    if args.epsilon:
        epsilons(args, data, device)

    if args.retrain_node:
        retrain_node(args, data, data['train_set'].nodes, device)
        loss_difference_node(args, data, device)

    if args.influence:
        edges_to_forget = sample_edges(args, data, method=args.method, num_edges=args.edges)
        print('The number of edges to compute infl is:', len(edges_to_forget))

        loss_diff = find_loss_difference(args, edges_to_forget)
        infls = influences(args, data, edges_to_forget, device)
        if args.save:
            df = pd.DataFrame({
                'influence': infls.values(),
                'loss_diff': loss_diff.values(),
            }, index=edges_to_forget)

            df.to_csv(os.path.join('./result', args.data, 'influence_vs_loss-diff.csv'))

        for e in edges_to_forget:
            print(f'Edge {e}, influence: {infls[e]:.4f}, loss_diff: {loss_diff[e]:.4f}.')

    if args.influence_node:
        # edges_to_forget = sample_edges(args, data, method=args.method, num_edges=args.edges)
        nodes_to_forget = sample_nodes(args, data, method=args.method, num_nodes=args.nodes)
        print('The number of edges to be forgotten is:', len(nodes_to_forget))

        loss_diff = find_loss_difference_node(args, nodes_to_forget)
        infls = influences_node(args, data, nodes_to_forget, device)
        if args.save:
            df = pd.DataFrame({
                'influence': infls.values(),
                'loss_diff': loss_diff.values(),
            }, index=edges_to_forget)

            df.to_csv(os.path.join('./result', args.data, 'node_influence_vs_loss-diff.csv'))

        for n, l in nodes_to_forget:
            print(f'node {n}, influence: {infls[(n, l)]:.4f}, loss_diff: {loss_diff[(n,l)]:.4f}.')

    if args.node_unlearn:
        edge_index = torch.tensor(data['edges'], device=device).t()
        test_loader = DataLoader(data['test_set'], batch_size=args.test_batch, shuffle=False)

        nodes_to_forget = sample_nodes(args, data)

        original_model = train_model(args, data, eval=False, verbose=False, device=device)

        original_res, _ = test(original_model, test_loader, edge_index, device)
        print(f'Original model: {original_res["accuracy"]:.4f}.')

        result = defaultdict(list)
        for num_nodes in tqdm(args.nodes, desc='retrain & unlearn'):
            _nodes = nodes_to_forget[:num_nodes]
            retrain_model, retrain_duration = retrain_node(args, data, _nodes, device)
            unlearn_model, unlearn_duration = node_unlearn(args, data, original_model, _nodes, device)

            edges_prime = edges_remove_nodes(data['edges'], _nodes)
            edge_index_prime = torch.tensor(edges_remove_nodes(data['edges'], _nodes), device=device).t()
            retrain_res, _ = test(retrain_model, test_loader, edge_index_prime, device)
            unlearn_res, _ = test(unlearn_model, test_loader, edge_index_prime, device)

            result['# edges'].append(num_nodes)
            result['setting'].append('RE')
            result['accuracy'].append(retrain_res['accuracy'])
            result['time'].append(retrain_duration)

            result['# edges'].append(num_nodes)
            result['setting'].append('UL')
            result['accuracy'].append(unlearn_res['accuracy'])
            result['time'].append(unlearn_duration)

        df = pd.DataFrame(result)
        print(df)
        df.to_csv(os.path.join('./result', f'node_unlearn_{args.model}_{args.data}.csv'))

    if args.unlearn:
        edges_to_forget = sample_edges(args, data, method=args.method)
        print(f'The number of edges ({args.method}) to be forgottn is:', args.edges)

        retrain_acc, retrain_f1, retrain_time = [], [], []
        retrain_loss = []
        for num_edges in tqdm(args.edges, desc='retrain'):
            # if args.method != 'random' and os.path.exists(model_path(args, 'retrain', edges=num_edges)):
            #     continue
            t0 = time.time()
            model_retrained, _ = retrain(args, data, edges_to_forget[:num_edges], device, forget_all=True)
            if args.save:
                save_model(args, model_retrained, type='retrain', edges=num_edges)
                print(f'Retrain duration: {(time.time() - t0): .4f}')
            else:
                test_loader = DataLoader(data['test_set'], batch_size=args.test_batch, shuffle=False)
                edges_ = [e for e in data['edges'] if e not in edges_to_forget[:num_edges]]
                edge_index_prime = torch.tensor(edges_, device=device).t()
                res, loss = test(model_retrained, test_loader, edge_index_prime, device)
                retrain_acc.append(res['accuracy'])
                retrain_f1.append(res['macro avg']['f1-score'])
                retrain_time.append(time.time() - t0)
                retrain_loss.append(loss)

        # baseline: GraphEraser
        baseline_time = defaultdict(list)
        baseline_acc = defaultdict(list)
        baseline_f1 = defaultdict(list)
        baseline_shared = defaultdict(list)
        for num_edges in tqdm(args.edges, desc='baseline'):
            for partition in ['blpa', 'bekm']:
                args.partition = partition
                args.aggr = aggr_method[(args.model, args.data, partition)]
                base_time, base_acc, base_f1, num_shards = graph_earser.unlearn(
                    args, data, edges_to_forget[:num_edges], device)
                baseline_time[partition].append(base_time)
                baseline_acc[partition].append(base_acc)
                baseline_f1[partition].append(base_f1)
                baseline_shared[partition].append(num_shards)

        if args.batch_unlearn:
            unlearn_acc, unlearn_f1, unlearn_loss, unlearn_time = batch_unlearn(
                args, data, edges_to_forget, args.edges, device)
        else:
            unlearn(args, data, edges_to_forget, args.edges, device)
        print('Result')
        for idx, num_edges in enumerate(args.edges):
            if idx % 3 == 0:
                print(f'  Retrain: Acc:{retrain_acc[idx]:.4f}, Running time: {retrain_time[idx]:.4f}.')
                print(f'  BLPA: Acc:{baseline_acc["blpa"][idx]:.4f}, Running time: {baseline_time["blpa"][idx]:.4f}.')
                print(f'  BEKM: Acc:{baseline_acc["bekm"][idx]:.4f}, Running time: {baseline_time["bekm"][idx]:.4f}.')
                print(f'  Unlearn: Acc:{unlearn_acc[idx]:.4f}, Running time: {unlearn_time[idx]:.4f}.')

        gnn_results_df = pd.DataFrame({
            'retrain-acc': retrain_acc,
            'retrain-time': retrain_time,
            'retrain-f1': retrain_f1,
            'unlearn-acc': unlearn_acc,
            'unlearn-time': unlearn_time,
            'unlearn-f1': unlearn_f1,
            'blpa-acc': baseline_acc['blpa'],
            'blpa-time': baseline_time['blpa'],
            'blpa-f1': baseline_f1['blpa'],
            'blpa-#-shards': baseline_shared['blpa'],
            'bekm-acc': baseline_acc['bekm'],
            'bekm-time': baseline_time['bekm'],
            'bekm-f1': baseline_f1['bekm'],
            'bekm-#-shards': baseline_shared['bekm'],
        }, index=args.edges)
        gnn_results_df.to_csv(f'{args.model}_{args.data}_rq1.csv')

    if args.rq1:
        edges_to_forget = sample_edges(args, data, method=args.method)
        print(f'The number of edges ({args.method}) to be forgottn is:', args.edges)

        # train a original model
        original_model = train_model(args, data, eval=False, verbose=False, device=device)

        test_loader = DataLoader(data['test_set'], batch_size=args.test_batch, shuffle=False)

        retrain_acc, unlearn_acc = [], []
        retrain_time, unlearn_time = [], []
        for num_edges in tqdm(args.edges, desc='RQ 1'):
            retrain_model, retrain_duration = retrain(args, data, edges_to_forget[:num_edges], device, forget_all=True)

            unlearn_model, unlearn_duration = unlearn(args, data, original_model, edges_to_forget[:num_edges], device)

            edges_ = [e for e in data['edges'] if e not in edges_to_forget[:num_edges]]
            edge_index_prime = torch.tensor(edges_, device=device).t()
            retrain_res, _ = test(retrain_model, test_loader, edge_index_prime, device)
            unlearn_res, _ = test(unlearn_model, test_loader, edge_index_prime, device)

            retrain_acc.append(retrain_res['accuracy'])
            retrain_time.append(retrain_duration)
            unlearn_acc.append(unlearn_res['accuracy'])
            unlearn_time.append(unlearn_duration)

        print('Result')
        for idx, num_edges in enumerate(args.edges):
            print(f'  {num_edges} edges:')
            print(f'    Retrain: Acc:{retrain_acc[idx]:.4f}, Running time: {retrain_time[idx]:.4f}.')
            print(f'    Unlearn: Acc:{unlearn_acc[idx]:.4f}, Running time: {unlearn_time[idx]:.4f}.')

        # baseline: GraphEraser

        # 1. original
        for partition in ['blpa', 'bekm']:
            args.partition = partition
            args.aggr = aggr_method[(args.model, args.data, partition)]
            acc, f1 = graph_earser.train(args, data, device)

        # 2. retrain
        baseline_time = defaultdict(list)
        baseline_acc = defaultdict(list)
        baseline_f1 = defaultdict(list)
        baseline_shared = defaultdict(list)
        for num_edges in tqdm(args.edges, desc='baseline'):
            for partition in ['blpa', 'bekm']:
                args.partition = partition
                args.aggr = aggr_method[(args.model, args.data, partition)]
                base_time, base_acc, base_f1, num_shards = graph_earser.unlearn(
                    args, data, edges_to_forget[:num_edges], device)
                baseline_time[partition].append(base_time)
                baseline_acc[partition].append(base_acc)
                baseline_f1[partition].append(base_f1)
                baseline_shared[partition].append(num_shards)

        print('Result')
        for idx, num_edges in enumerate(args.edges):
            if idx % 3 == 0:
                print(f'  Retrain: Acc:{retrain_acc[idx]:.4f}, Running time: {retrain_time[idx]:.4f}.')
                print(f'  BLPA: Acc:{baseline_acc["blpa"][idx]:.4f}, Running time: {baseline_time["blpa"][idx]:.4f}.')
                print(f'  BEKM: Acc:{baseline_acc["bekm"][idx]:.4f}, Running time: {baseline_time["bekm"][idx]:.4f}.')
                print(f'  Unlearn: Acc:{unlearn_acc[idx]:.4f}, Running time: {unlearn_time[idx]:.4f}.')

        gnn_results_df = pd.DataFrame({
            'retrain-acc': retrain_acc,
            'retrain-time': retrain_time,
            # 'retrain-f1': retrain_f1,
            'unlearn-acc': unlearn_acc,
            'unlearn-time': unlearn_time,
            # 'unlearn-f1': unlearn_f1,
            'blpa-acc': baseline_acc['blpa'],
            'blpa-time': baseline_time['blpa'],
            # 'blpa-f1': baseline_f1['blpa'],
            'blpa-#-shards': baseline_shared['blpa'],
            'bekm-acc': baseline_acc['bekm'],
            'bekm-time': baseline_time['bekm'],
            # 'bekm-f1': baseline_f1['bekm'],
            'bekm-#-shards': baseline_shared['bekm'],
        }, index=args.edges)
        gnn_results_df.to_csv(f'{args.model}_{args.data}_rq1.csv')

    if args.rq2:
        hiddens = {
            'RI': [[], [16], [24, 12], [24, 16, 8]],
            'NF': [[], [16], [32, 16], [64, 32, 16]]
        }
        damping = {
            'RI': {
                'gcn': [0.1, 1, 10, 100],
                'gat': [100, 1000, 1000, 1000],
                'sage': [1, 100, 100, 100],
            },
            'NF': {
                'gcn': [1, 10, 100, 100],
                'gat': [100, 1000, 1000, 1000],
                'sage': [1, 100, 100, 100],
            },
        }

        num_edges = 1000
        test_loader = DataLoader(data['test_set'], batch_size=args.test_batch, shuffle=False)
        edge_index = torch.tensor(data['edges'], device=device).t()

        d = defaultdict(list)
        for _ in tqdm(range(10), desc='RQ2'):
            edges_to_forget = sample_edges(args, data, method=args.method)[:num_edges]
            for i in range(4):
                for setting in ['RI', 'NF']:
                    args.hidden = hiddens[setting][i]
                    args.feature = setting == 'NF'
                    args.no_feature_update = setting == 'RI'

                    # train a original model
                    original_model = train_model(args, data, eval=False, verbose=False, device=device)
                    # retrain
                    retrain_model, _ = retrain(args, data, edges_to_forget, device, forget_all=True)
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
        df.to_csv(os.path.join('./result', f'{args.model}_{args.data}_rq2.csv'))

    if args.adv:
        adversaracy_setting(args, data, device)

    if args.rq4:
        args.edges = [100, 500, 1000, 1500, 2000]
        adv_origianl_acc, adv_retrain_acc, adv_unlearn_acc = adversaracy_setting(args, data, device)
        df = pd.DataFrame({
            '# edges': args.edges + args.edges + args.edges,
            'setting': ['Adversarial'] * len(args.edges) + ['Benign'] * len(args.edges) + ['Unlearn'] * len(args.edges),
            'Accuracy': adv_origianl_acc + adv_retrain_acc + adv_unlearn_acc,
        })
        df.to_csv(os.path.join('./result', f'{args.model}_{args.data}_rq4_unlearn.csv'))

        edges_to_forget = sample_edges(args, data, method='random')
        retrain_acc = []
        for num_edges in tqdm(args.edges, desc='retrain'):
            model_retrained, _ = retrain(args, data, edges_to_forget[:num_edges], device, forget_all=True)
            test_loader = DataLoader(data['test_set'], batch_size=args.test_batch, shuffle=False)
            edges_ = [e for e in data['edges'] if e not in edges_to_forget[:num_edges]]
            edge_index_prime = torch.tensor(edges_, device=device).t()
            res, loss = test(model_retrained, test_loader, edge_index_prime, device)
            retrain_acc.append(res['accuracy'])

        unlearn_acc, _, _, _ = batch_unlearn(args, data, edges_to_forget, args.edges, device)

        bengin_diff = np.abs(np.array(retrain_acc) - np.array(unlearn_acc))
        adv_diff = np.abs(np.array(adv_retrain_acc) - np.array(adv_unlearn_acc))

        df = pd.DataFrame({
            '# edges': args.edges + args.edges,
            'type': ['Benign'] * len(args.edges) + ['Adversarial'] * len(args.edges),
            'Difference': np.concatenate((bengin_diff, adv_diff))
        })
        df.to_csv(os.path.join('./result', f'{args.model}_{args.data}_rq4_diff.csv'))

    if args.mia_attack:
        num_edges = args.edges[0]

        # prepare data
        test_loader = DataLoader(data['test_set'], batch_size=1024, shuffle=False)
        edge_index = torch.tensor(data['edges'], device=device).t()

        original_model = train_model(args, data, eval=False, verbose=False, device=device)
        orig_res, _ = test(original_model, test_loader, edge_index, device)
        orig_acc = orig_res['accuracy']
        print(f'The accuracy of original model is: {orig_acc:.4f}.')

        results = {}
        for method in ['random', 'same_class', 'diff_class']:
            edges_to_forget = sample_edges(args, data, method=method)[:num_edges]
            # two_hop_edges = []
            # for v1, v2 in tqdm(edges_to_forget, '2-hop:'):
            #     for e in data['edges']:
            #         if e in two_hop_edges or (e[1], e[0]) in two_hop_edges:
            #             continue
            #         if v1 == e[0] and ((e[1], v2) in data['edges'] or (v2, e[1]) in data['edges']):
            #             two_hop_edges.append(e)
            #         if v1 == e[1] and ((e[0], v2) in data['edges'] or (v2, e[0]) in data['edges']):
            #             two_hop_edges.append(e)

            # continue

            edges_prime = [e for e in data['edges'] if e not in edges_to_forget]
            # edges_prime_2hop = [e for e in edges_prime if e not in two_hop_edges]

            retrain_model, _ = retrain(args, data, edges_to_forget, device, forget_all=True)
            unlearn_model, _ = unlearn(args, data, original_model, edges_to_forget, device)

            edge_index_prime = torch.tensor(edges_prime, device=device).t()
            retrain_res, _ = test(retrain_model, test_loader, edge_index_prime, device)
            unlearn_res, _ = test(unlearn_model, test_loader, edge_index_prime, device)
            retrain_acc = retrain_res['accuracy']
            unlearn_acc = unlearn_res['accuracy']
            print(
                f'Finished retrain and unlear, the accuracy are: retrain({retrain_acc:.4f} and unlearn({unlearn_acc:.4f}).')

            # MIA attak
            mia_edges, mia_labels = construct_mia_data(data, edges_prime, edges_to_forget)

            # train MIA on original GNN
            original_model.eval()
            with torch.no_grad():
                y_hat = original_model(data['nodes'], edge_index)
            posterior_o = F.softmax(y_hat, dim=1)
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
            x_train_r, y_train_r, x_test_r, y_test_r = generate_mia_features(mia_edges, mia_labels,
                                                                             sample_member(
                                                                                 edges_prime, edges_to_forget, num_edges) + edges_to_forget,
                                                                             [1] * num_edges + [0] * num_edges,
                                                                             posterior_r.cpu().numpy())
            mia = MIA(x_train_r, y_train_r)
            r_mia_acc, r_mia_tp, r_mia_tn = evaluate_mia(mia, x_test_r, y_test_r)

            # test MIA on unlearned GNN
            unlearn_model.eval()
            with torch.no_grad():
                y_hat = unlearn_model(data['nodes'], edge_index_prime)
            posterior_u = F.softmax(y_hat, dim=1)
            x_train_u, y_train_u, x_test_u, y_test_u = generate_mia_features(mia_edges, mia_labels,
                                                                             sample_member(
                                                                                 edges_prime, edges_to_forget, num_edges) + edges_to_forget,
                                                                             [1] * num_edges + [0] * num_edges,
                                                                             posterior_u.cpu().numpy())
            mia = MIA(x_train_u, y_train_u)
            u_mia_acc, u_mia_tp, u_mia_tn = evaluate_mia(mia, x_test_u, y_test_u)

            # orig_mia_acc, orig_tp, orig_fn = mia_attack(
            #     original_model,
            #     data['nodes'], edge_index,
            #     mia_edges, mia_labels,
            #     sample_non_member(data['num_nodes'], data['edges'], num_edges) + edges_to_forget,
            #     [0] * num_edges + [1] * num_edges,
            #     features=data['features']
            # )

            # retrain_mia_acc, r_tp, r_fn = mia_attack(
            #     retrain_model,
            #     data['nodes'], edge_index_prime,
            #     mia_edges, mia_labels,
            #     sample_member(edges_prime, num_edges) + edges_to_forget,
            #     [1] * num_edges + [0] * num_edges,
            #     features=data['features']
            # )

            # unlearn_mia_acc, u_tp, u_fn = mia_attack(
            #     unlearn_model,
            #     data['nodes'], edge_index_prime,
            #     mia_edges, mia_labels,
            #     sample_member(edges_prime, num_edges) + edges_to_forget,
            #     [1] * num_edges + [0] * num_edges,
            #     features=data['features']
            # )

            results[method] = {
                'original': [o_mia_acc, o_mia_tp, o_mia_tn],
                'retrain': [r_mia_acc, r_mia_tp, r_mia_tn],
                'unlearn': [u_mia_acc, u_mia_tp, u_mia_tn],
            }

        print(results)
