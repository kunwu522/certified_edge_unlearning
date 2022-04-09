import os
import pickle
import random
import numpy as np
import torch
from collections import defaultdict
from model import GNN

# percentage_num_edges = {
#     0.1: 3,
#     1: 50,
#     10: 500,
#     30: 1500,
#     50: 2000,
# }


def model_path(args, type, edges=None):
    if args.hidden:
        layers = '-'.join([str(h) for h in args.hidden])
        prefix = f'{args.model}_{args.data}_{layers}'
    else:
        prefix = f'{args.model}_{args.data}'

    if type == 'original':
        return os.path.join('./checkpoint', args.data, f'{prefix}_best.pt')
    elif type == 'retrain':
        if args.max_degree:
            filename = f'{prefix}_{type}_max_{args.method}{edges}_best.pt'
        else:
            filename = f'{prefix}_{type}_{args.method}{edges}_best.pt'
        return os.path.join('./checkpoint', args.data, filename)
    elif type == 'unlearn':
        assert edges is not None
        if args.batch_unlearn:
            prefix += '_batch'
        if args.unlearn_batch_size is not None:
            prefix += f'args.unlearn_batch_size'
        if args.approx == 'lissa':
            filename = f'{prefix}_{type}_{args.method}{edges}_{args.approx}d{args.depth}r{args.r}_best.pt'
        else:
            filename = f'{prefix}_{type}_{args.method}{edges}_{args.approx}_d{args.damping}_best.pt'
        return os.path.join('./checkpoint', args.data, filename)
    else:
        raise ValueError('Invalid type of model,', type)


def create_model(args, data):
    embedding_size = args.emb_dim if data['features'] is None else data['features'].shape[1]
    model = GNN(data['num_nodes'], embedding_size,
                args.hidden, data['num_classes'], data['features'], args.feature_update, args.model)
    return model


def load_model(args, data, type='original', edges=None, edge=None, node=None):
    assert type in ['original', 'edge', 'node', 'retrain', 'unlearn'], f'Invalid type of model, {type}'
    if type == 'edge':
        model = create_model(args, data)
        model.load_state_dict(torch.load(os.path.join('./checkpoint', args.data, 'edges',
                              f'{args.model}_{args.data}_{edge[0]}_{edge[1]}_best.pt')))
        return model
    elif type == 'node':
        model = create_model(args, data)
        model.load_state_dict(torch.load(os.path.join('./checkpoint', args.data, 'nodes',
                              f'{args.model}_{args.data}_{node}_best.pt')))
        return model
    else:
        model = create_model(args, data)
        model.load_state_dict(torch.load(model_path(args, type, edges)))
        return model


def save_model(args, model, type='original', edges=None, edge=None, node=None):
    assert type in ['original', 'edge', 'node', 'retrain', 'unlearn'], f'Invalid type of model, {type}'

    if type == 'edge':
        torch.save(model.state_dict(), os.path.join('./checkpoint', args.data, 'edges'
                   f'{args.model}_edges_{args.data}_{edge[0]}_{edge[1]}_best.pt'))
    elif type == 'node':
        torch.save(model.state_dict(), os.path.join('./checkpoint', args.data, 'nodes'
                   f'{args.model}_nodes_{args.data}_{node}_best.pt'))
    else:
        print('save model to', model_path(args, type, edges))
        torch.save(model.state_dict(), model_path(args, type, edges))


def sample_edges(args, data, method='random'):
    if method == 'random':
        edges_to_forget = [e for e in data['edges']]
        random.shuffle(edges_to_forget)
    elif method == 'degree':
        node_degree = defaultdict(int)
        for edge in data['edges']:
            node_degree[edge[0]] += 1
            node_degree[edge[1]] += 1
        edge_degree = {(e[0], e[1]): node_degree[e[0]] + node_degree[e[1]] for e in data['edges']}
        sorted_edge_degree = {k: v for k, v in sorted(
            edge_degree.items(), key=lambda item: item[1], reverse=args.max_degree)}
        edges_to_forget = list(sorted_edge_degree.keys())
    elif method == 'loss_diff':
        edge2loss_diff = find_loss_difference(args, data['edges'])
        sorted_edge2loss_diff = {k: v for k, v in sorted(edge2loss_diff.items(), key=lambda x: abs(x[1]), reverse=True)}
        edges_to_forget = list(sorted_edge2loss_diff.keys())
    else:
        raise ValueError(f'Invalid sample method: {method}.')

    return edges_to_forget


def sample_nodes(args, data, method='loss_diff', num_nodes=-1):
    n = int(num_nodes * 0.01 * len(data['train_set'])) if num_nodes != -1 else 1

    if method == 'random':
        indices = list(range(len(data['train_set'])))
        random.shuffle(indices)
        sampled_nodes = np.array(data['train_set'].nodes)[indices]
        sampled_labels = np.array(data['train_set'].labels)[indices]
    elif method == 'degree':
        raise NotImplementedError()
    elif method == 'loss_diff':
        node2loss_diff = find_loss_difference_node(
            args, [(n, l) for n, l in zip(data['train_set'].nodes, data['train_set'].labels)])
        sorted_node2loss_diff = {k: v for k, v in sorted(node2loss_diff.items(), key=lambda x: abs(x[1]), reverse=True)}
        sampled = np.array(list(sorted_node2loss_diff.keys()))
        sampled_nodes = sampled[:, 0]
        sampled_labels = sampled[:, 1]

    if num_nodes == -1:
        return sampled_nodes[0], sampled_labels[0]
    else:
        return sampled_nodes[:n], sampled_labels[:n]


def loss_of_test_nodes(args, data, device=torch.device('cpu')):
    model = load_model(args, data).to(device)

    test_set = data['test_set']
    edge_index = torch.tensor(data['edges'], device=device).t()
    model.eval()
    with torch.no_grad():
        y_hat = model(torch.tensor(test_set.nodes, device=device), edge_index)
        test_losses = model.losses(y_hat, torch.tensor(test_set.labels, device=device))

    test_node_loss = []
    for node, label, loss in zip(test_set.nodes, test_set.labels, test_losses.cpu().tolist()):
        test_node_loss.append({
            'node': node,
            'label': label,
            'loss': loss,
        })

    return sorted(test_node_loss, key=lambda x: x['loss'], reverse=True)


def find_loss_difference(args, edges_to_forget):
    loss_diff_path = os.path.join('./data', args.data, 'edge_loss_difference.dict')
    if not os.path.exists(loss_diff_path):
        raise FileExistsError('Could not find loss difference results. Please use -loss-diff to generate it first.')

    with open(loss_diff_path, 'rb') as fp:
        edge2loss_diff = pickle.load(fp)

    result = {e: edge2loss_diff[e] for e in edges_to_forget}
    return result


def find_loss_difference_node(args, nodes_to_forget):
    loss_diff_path = os.path.join('./data', args.data, 'node_loss_difference.dict')
    if not os.path.exists(loss_diff_path):
        raise FileExistsError('Could not find loss difference results. Please use -loss-diff to generate it first.')

    with open(loss_diff_path, 'rb') as fp:
        edge2loss_diff = pickle.load(fp)

    result = {(n, l): edge2loss_diff[(n, l)] for n, l in nodes_to_forget}
    return result
