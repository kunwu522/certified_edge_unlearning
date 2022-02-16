import os
import copy
import pickle
import torch
from tqdm import tqdm
from train import train_gcn
from utils import load_model, loss_of_test_nodes


def retrain(args, data, edges_to_forget, device, forget_all=False):
    print('Start to retrain...')

    if forget_all:
        edges_ = [(v1, v2) for v1, v2 in data['edges'] if (v1, v2) not in edges_to_forget]
        data_ = copy.deepcopy(data)
        data_['edges'] = edges_
        retrain_model = train_gcn(args, data_, eval=False, device=device, verbose=False)
        print('Retraining finished.')
        print()
        return retrain_model

    edges_ = [(v1, v2) for v1, v2 in data['edges']]
    for edge in tqdm(edges_to_forget, desc='  retraining'):
        v1, v2 = edge
        edges_.remove((v1, v2))
        data_ = copy.deepcopy(data)
        data_['edges'] = edges_
        retrain_model = train_gcn(args, data_, eval=False, device=device, verbose=False)
        torch.save(retrain_model.state_dict(), os.path.join(
            './checkpoint', 'all', f'gcn_{args.data}_{v1}_{v2}_best.pt'))
        edges_ = [(v1, v2) for v1, v2 in data['edges']]

    print('Retraining finished.')
    print()


def loss_difference(args, data, device):
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

        edges_ = [(v1, v2) for v1, v2 in data['edges']]

        for edge in tqdm(data['edges'], desc='  loss diff'):
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
