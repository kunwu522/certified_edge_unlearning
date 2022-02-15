import os
import pickle
import torch
import random

from model import GCN


def load_model(args, data, type='original', edge=None):
    assert type in ['original', 'edge', 'retrain', 'unlearn'], f'Invalid type of model, {type}'

    if type == 'original':
        embedding_size = args.emb_dim if data['features'] is None else data['features'].shape[1]
        model = GCN(data['num_nodes'], embedding_size,
                    args.hidden, data['num_classes'], data['features'])
        model.load_state_dict(torch.load(os.path.join('./checkpoint', f'gcn_{args.data}_best.pt')))
        return model
    elif type == 'edge':
        embedding_size = args.emb_dim if data['features'] is None else data['features'].shape[1]
        model = GCN(data['num_nodes'], embedding_size,
                    args.hidden, data['num_classes'], data['features'])
        model.load_state_dict(torch.load(os.path.join('./checkpoint', f'gcn_{args.data}_{edge[0]}_{edge[1]}_best.pt')))
        return model
    else:
        embedding_size = args.emb_dim if data['features'] is None else data['features'].shape[1]
        model = GCN(data['num_nodes'], embedding_size,
                    args.hidden, data['num_classes'], data['features'])
        model.load_state_dict(torch.load(os.path.join('./checkpoint', f'gcn_{args.data}_{type}_best.pt')))


def save_model(args, model, type='original', edge=None):
    assert type in ['original', 'edge', 'retrain', 'unlearn'], f'Invalid type of model, {type}'

    if type == 'original':
        torch.save(model.state_dict(), os.path.join('./checkpoint', f'gcn_{args.data}_best.pt'))
    elif type == 'edge':
        torch.save(model.state_dict(), os.path.join('./checkpoint', f'gcn_{args.data}_{edge[0]}_{edge[1]}_best.pt'))
    else:
        torch.save(model.state_dict(), os.path.join('./checkpoint', f'gcn_{args.data}_{type}_best.pt'))


def sample_edges(args, data, method='random'):
    edges_path = os.path.join('./data', args.data, 'edges_to_forget_{method}.list')
    if os.path.exists(edges_path):
        with open(edges_path, 'rb') as fp:
            edges_to_forget = pickle.load(fp)
        return edges_to_forget

    if method == 'random':
        num_edges = int(args.num_edges * 0.01 * len(data['edges']))
        edges_to_forget = random.sample(data['edges'], num_edges)
    elif method == 'degree':
        raise NotImplementedError(f'{method} did not implement yet.')
    elif method == 'loss_diff':
        raise NotImplementedError(f'{method} did not implement yet.')
    else:
        raise ValueError(f'Invalid sample method: {method}.')

    with open(edges_path, 'wb') as fp:
        pickle.dump(edges_to_forget, fp)

    return edges_to_forget


def sample_nodes(dataset):
    random_index = random.choice(range(len(dataset.nodes)))
    sampled_node = dataset.nodes[random_index]
    sampled_label = dataset.labels[random_index]
    return sampled_node, sampled_label


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
