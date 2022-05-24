import os
import copy
import time
import pickle
import torch
import torch.nn.functional as F
from tqdm import tqdm
from train import train_model
from utils import load_model, loss_of_test_nodes, remove_undirected_edges, train_set_remove_nodes, edges_remove_nodes


def retrain(args, data, edges_to_forget, device, verbose=False, return_epoch=False):
    _edges = remove_undirected_edges(data['edges'], edges_to_forget)
    _data = copy.deepcopy(data)
    _data['edges'] = _edges
    # print(f'Original number of edges {data["num_edges"]}, retrain with {len(data_["edges"])}.')
    t0 = time.time()
    if return_epoch:
        retrain_model, num_epochs = train_model(
            args, _data, eval=False, device=device, verbose=verbose, return_epoch=return_epoch)
    else:
        retrain_model = train_model(args, _data, eval=False, device=device, verbose=verbose, return_epoch=return_epoch)
    duration = time.time() - t0
    if return_epoch:
        return retrain_model, duration, num_epochs
    else:
        return retrain_model, duration


def loss_difference(args, data, edges, device):
    test_node = loss_of_test_nodes(args, data, device=device)[0]

    loss_diff_path = os.path.join('./data', args.data, 'edge_loss_difference.dict')
    if os.path.exists(loss_diff_path):
        with open(loss_diff_path, 'rb') as fp:
            result = pickle.load(fp)
    else:
        result = {}

        node = torch.tensor(test_node['node'], device=device)
        label = torch.tensor(test_node['label'], device=device)
        edge_index = torch.tensor(data['edges'], device=device).t()

        original_model = load_model(args, data).to(device)
        original_model.eval()
        with torch.no_grad():
            y_hat = original_model(node, edge_index).unsqueeze(0)
            original_loss = original_model.loss(y_hat, label.unsqueeze(0))

        edges_ = [(v1, v2) for v1, v2 in edges]

        for edge in tqdm(edges, desc='  loss diff'):
            edges_.remove(edge)
            retrain_edge_index = torch.tensor(edges_, dtype=torch.long, device=device).t()

            retrain_model = load_model(args, data, type='edge', edge=edge).to(device)
            retrain_model.eval()
            with torch.no_grad():
                y_hat = retrain_model(node, retrain_edge_index)
                retrain_loss = retrain_model.loss(y_hat.unsqueeze(0), label.unsqueeze(0))

            result[edge] = (retrain_loss - original_loss).detach().cpu().item()
            edges_ = [(v1, v2) for v1, v2 in data['edges']]

            del retrain_model
            torch.cuda.empty_cache()

        with open(loss_diff_path, 'wb') as fp:
            pickle.dump(result, fp)

    return result


def retrain_node(args, data, nodes_to_forget, device):
    _data = train_set_remove_nodes(data, nodes_to_forget)
    _data['edges'] = edges_remove_nodes(data['edges'], nodes_to_forget)
    t0 = time.time()
    retrain_model = train_model(args, _data, eval=False, device=device, verbose=False)
    return retrain_model, time.time() - t0


def loss_difference_node(args, data, device):
    test_node = loss_of_test_nodes(args, data, device=device)[0]

    loss_diff_path = os.path.join('./data', args.data, 'node_loss_difference.dict')
    if os.path.exists(loss_diff_path):
        with open(loss_diff_path, 'rb') as fp:
            result = pickle.load(fp)
    else:
        result = {}

        x_test = torch.tensor([test_node['node']], device=device)
        y_test = torch.tensor([test_node['label']], device=device)
        edge_index = torch.tensor(data['edges'], device=device).t()

        original_model = load_model(args, data).to(device)
        original_model.eval()
        with torch.no_grad():
            y_hat = original_model(x_test, edge_index)
            original_loss = original_model.loss(y_hat, y_test)

        for node, label in tqdm(data['train_set'], desc='  loss diff'):
            retrain_model = load_model(args, data, type='node', node=node).to(device)
            retrain_model.eval()
            with torch.no_grad():
                y_hat = retrain_model(x_test, edge_index)
                retrain_loss = retrain_model.loss(y_hat, y_test)

            result[(node, label)] = (retrain_loss - original_loss).detach().cpu().item()

            del retrain_model
            torch.cuda.empty_cache()

        with open(loss_diff_path, 'wb') as fp:
            pickle.dump(result, fp)

    return result


def epsilons(args, data, device):
    epsilons_path = os.path.join('./data', args.data, 'edge_epsilon.dict')
    if os.path.exists(epsilons_path):
        with open(epsilons_path, 'rb') as fp:
            edge2epsilon = pickle.load(fp)
        return edge2epsilon

    model = load_model(args, data).to(device)
    nodes = torch.tensor(data['train_set'].nodes, device=device)
    labels = torch.tensor(data['train_set'].labels, device=device)
    edge_index = torch.tensor(data['edges'], device=device).t()

    model.eval()
    with torch.no_grad():
        y_hat = model(nodes, edge_index)
        losses = model.losses(y_hat, labels)

    edge2epsilon = {}
    for edge in tqdm(data['edges'], desc='calculating epsilon'):
        retrained_model = load_model(args, data, type='edge', edge=edge).to(device)
        _edges = [e for e in data['edges'] if e != edge]
        assert len(_edges) == len(data['edges']) - 1, f'The number of new edges is not right. {len(_edges)}'
        edge_index_prime = torch.tensor(_edges, device=device).t()

        # For embedding or output of GCN
        # with torch.no_grad():
        #     y_hat_prime = F.softmax(model(nodes, edge_index_prime), dim=1)
        # num_nonzero = torch.unique(torch.nonzero(y_hat_prime - y_hat)[:, 0]).cpu().size(0)

        # The number of nodes that loss got changed.
        with torch.no_grad():
            y_hat_prime = retrained_model(nodes, edge_index_prime)
            losses_prime = retrained_model.losses(y_hat_prime, labels)
        diff = torch.abs(losses_prime - losses)
        num_nonzero = len(torch.where(diff > 0.01)[0])
        edge2epsilon[edge] = num_nonzero

    with open(epsilons_path, 'wb') as fp:
        pickle.dump(edge2epsilon, fp)

    print('results:')
    print(edge2epsilon)
