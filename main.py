import os
import time
import random
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from torch.utils.data import DataLoader
from adversarial_attack import adversaracy_setting
from argument import get_args
from data_loader import load_data
from train import train_model, test
from retrain import loss_difference_node, retrain, loss_difference, retrain_node, epsilons
from unlearn import batch_unlearn, influences_node, unlearn, influences
from utils import create_model, find_loss_difference_node, model_path, save_model, sample_edges, find_loss_difference, sample_nodes


if __name__ == '__main__':
    args = get_args()
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

    if args.unlearn:
        edges_to_forget = sample_edges(args, data, method=args.method)
        print(f'The number of edges ({args.method}) to be forgottn is:', args.edges)

        retrain_acc, retrain_f1, retrain_time = [], [], []
        for num_edges in tqdm(args.edges, desc='retrain'):
            # if args.method != 'random' and os.path.exists(model_path(args, 'retrain', edges=num_edges)):
            #     continue
            t0 = time.time()
            model_retrained = retrain(args, data, edges_to_forget[:num_edges], device, forget_all=True)
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

        if args.batch_unlearn:
            unlearn_acc, unlearn_f1, unlearn_time = batch_unlearn(args, data, edges_to_forget, args.edges, device)
        else:
            unlearn(args, data, edges_to_forget, args.edges, device)

        print('Result:')
        print('  Retrain accuracy:', retrain_acc)
        print('  Retrain F1:', retrain_f1)
        print('  unlearn accuracy:', unlearn_acc)
        print('  unlearn F1:', unlearn_f1)

        print('  retrain time:', retrain_time)
        print('  unlearn time:', unlearn_time)

    if args.adv:
        adversaracy_setting(args, data, device)
