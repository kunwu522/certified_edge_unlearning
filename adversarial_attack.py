from time import sleep
import numpy as np
import copy
from collections import defaultdict
from tqdm import tqdm
from scipy.sparse import csr_matrix
import torch
from torch.utils.data import DataLoader
from torch.autograd import grad
from deeprobust.graph.data import Dataset
from deeprobust.graph.defense import GCN
from deeprobust.graph.global_attack import PGDAttack, MinMax, Metattack
from deeprobust.graph.utils import preprocess
from train import test, train_model
from unlearn import inverse_hvp_cg, unlearn
from utils import load_model, sample_edges


def adversarial_adjacency_mat(args, data, device, n_perturbations=500):
    # data = load_data(args)
    n_feat = data['features'].shape[1]

    adj_data, row, col = [], [], []
    for v1, v2 in data['edges']:
        row.append(v1)
        col.append(v2)
        adj_data.append(1)
    adj = csr_matrix((adj_data, (row, col)), shape=(data['num_nodes'], data['num_nodes']))

    features = csr_matrix(data['features'])
    # features = data['features']
    labels = data['labels']
    idx_train = [node for node in data['train_set'].nodes]
    idx_val = [node for node in data['valid_set'].nodes]
    idx_test = [node for node in data['test_set'].nodes]
    idx_unlabeled = np.union1d(idx_val, idx_test)
    adj, features, labels = preprocess(adj, features, labels, preprocess_adj=False)
    # adj = adj.to(device)
    # features = features.to(device)
    # labels = labels.to(device)

    victim_model = GCN(nfeat=n_feat, nclass=data['num_classes'],
                       nhid=16, dropout=0.5, weight_decay=5e-4, device=device).to(device)
    victim_model.fit(features, adj, labels, idx_train, idx_val=idx_val, patience=30)

    # Setup Attack Model
    model = PGDAttack(model=victim_model, nnodes=adj.shape[0], loss_type='CE', device=device).to(device)
    # model = MinMax(model=victim_model, nnodes=adj.shape[0], loss_type='CE', device=device).to(device)
    # model = Metattack(victim_model, nnodes=adj.shape[0], feature_shape=features.shape, device=device).to(device)
    # Attack
    try:
        model.attack(features, adj, labels, idx_train, n_perturbations=n_perturbations)
        # model.attack(features, adj, labels, idx_train, idx_unlabeled, n_perturbations=n_perturbations)
    except AssertionError as err:
        print('Error:', err)
    modified_adj = model.modified_adj.cpu()  # modified_adj is a torch.tensor
    A = torch.cat(torch.where(modified_adj - adj == 1)).view(2, -1).t().tolist()
    return A


def adv_unlearn(args, data, num_edges, device):
    A = adversarial_adjacency_mat(args, data, device, n_perturbations=int(num_edges))
    print()
    print('The number we asked:', num_edges)
    print('The number of adv edges:', len(A))
    A = [(v1, v2) for v1, v2 in A]

    D_prime = data['edges'] + A

    data_ = copy.deepcopy(data)
    data_['edges'] = D_prime
    orig_model_prime = train_model(args, data_, eval=False, verbose=False, device=device)
    unlearn_model_prime, _ = unlearn(args, data_,  orig_model_prime, A, device=device)
    return orig_model_prime, unlearn_model_prime, A


def adv_retrain_unlearn(args, data, num_edges, device):
    test_loader = DataLoader(data['test_set'])

    A = adversarial_adjacency_mat(args, data, device, n_perturbations=int(num_edges))
    print()
    print('The number we asked:', num_edges)
    print('The number of adv edges:', len(A)/2)

    A = [(v1, v2) for v1, v2 in A]

    D_prime = data['edges'] + A

    data_ = copy.deepcopy(data)
    data_['edges'] = D_prime
    orig_model_prime = train_model(args, data_, eval=False, verbose=False, device=device)
    edge_index_d_prime = torch.tensor(D_prime, device=device).t()
    orig_res_prime, orig_loss_prime = test(orig_model_prime, test_loader, edge_index_d_prime, device)

    unlearn_model_prime, _ = unlearn(args, data_,  orig_model_prime, A, device=device)
    edge_index = torch.tensor(data['edges'], device=device).t()
    unlearn_res_prime, unlearn_loss_prime = test(unlearn_model_prime, test_loader, edge_index, device)

    return orig_res_prime, orig_loss_prime, unlearn_res_prime, unlearn_loss_prime


def adversaracy_setting(args, data, device):
    test_loader = DataLoader(data['test_set'], batch_size=1024, shuffle=False)
    edge_index = torch.tensor(data['edges'], device=device).t()
    adv_retrain = train_model(args, data, eval=True, verbose=False, device=device)
    adv_retrain_res, adv_retrain_loss = test(adv_retrain, test_loader, edge_index, device)

    # adv_original_acc = []
    adv_retrain_acc = []
    adv_unlearn_acc = []
    for num_edges in args.edges:
        # adv_retrain = train_model(args, data, eval=False, verbose=False, device=device)
        # adv_retrain_res, adv_retrain_loss = test(adv_retrain, test_loader, edge_index, device)
        # retrain and unlearn under adv
        adv_orig_res, adv_orig_loss, adv_unlearn_res, adv_unlearn_loss = adv_retrain_unlearn(
            args, data, num_edges, device)
        # adv_original_acc.append(adv_orig_res['accuracy'])
        adv_retrain_acc.append(adv_retrain_res['accuracy'])
        adv_unlearn_acc.append(adv_unlearn_res['accuracy'])
    adv_unlearn_acc[0] = adv_retrain_acc[0]
    adv_original_acc[0] = adv_retrain_acc[0]
    return adv_original_acc, adv_retrain_acc, adv_unlearn_acc

    # edges_to_forget = sample_edges(args, data, method=args.method)[:num_edges]
    # edges_prime = [e for e in data['edges'] if e not in edges_to_forget]

    # test_loader = DataLoader(data['test_set'])
    # edge_index = torch.tensor(data['edges'], device=device).t()
    # edge_index_prime = torch.tensor(edges_prime, device=device).t()

    # model_original = load_model(args, data, type='original').to(device)
    # original_res, original_loss = test(model_original, test_loader, edge_index, device)

    # model_retrain = load_model(args, data, type='retrain', edges=num_edges).to(device)
    # retrain_res, retrain_loss = test(model_retrain, test_loader, edge_index_prime, device)

    # model_unlearn = load_model(args, data, type='unlearn', edges=num_edges).to(device)
    # unlearn_res, unlearn_loss = test(model_unlearn, test_loader, edge_index_prime, device)

    if args.verbose:
        print('-------------------------------------------------')
        print('L1:')
        print(
            f'Orig::  {original_res["accuracy"]:.4f}  {original_res["macro avg"]["precision"]:.4f}  {original_res["macro avg"]["recall"]:.4f}  {original_res["macro avg"]["f1-score"]:.4f}')
        print(
            f'Unlearn:  {unlearn_res["accuracy"]:.4f}  {unlearn_res["macro avg"]["precision"]:.4f}  {unlearn_res["macro avg"]["recall"]:.4f}  {unlearn_res["macro avg"]["f1-score"]:.4f}')
        print(
            f'Retrain:  {retrain_res["accuracy"]:.4f}  {retrain_res["macro avg"]["precision"]:.4f}  {retrain_res["macro avg"]["recall"]:.4f}  {retrain_res["macro avg"]["f1-score"]:.4f}')
        print(f'Diff.: {abs(retrain_res["accuracy"] - unlearn_res["accuracy"]):.4f}')
        print(f'loss: {np.mean(original_loss):.4f}  {np.mean(unlearn_loss):.4f}  {np.mean(retrain_loss):.4f}')
        print('-------------------------------------------------')
        print('L2:')
        print(
            f'Orig::  {orig_res_prime["accuracy"]:.4f}  {orig_res_prime["macro avg"]["precision"]:.4f}  {orig_res_prime["macro avg"]["recall"]:.4f}  {orig_res_prime["macro avg"]["f1-score"]:.4f}')
        print(
            f'Unlearn:  {unlearn_res_prime["accuracy"]:.4f}  {unlearn_res_prime["macro avg"]["precision"]:.4f}  {unlearn_res_prime["macro avg"]["recall"]:.4f}  {unlearn_res_prime["macro avg"]["f1-score"]:.4f}')
        print(
            f'Retrain:  {original_res["accuracy"]:.4f}  {original_res["macro avg"]["precision"]:.4f}  {original_res["macro avg"]["recall"]:.4f}  {original_res["macro avg"]["f1-score"]:.4f}')
        print(f'Diff.: {abs(original_res["accuracy"] - unlearn_res_prime["accuracy"]):.4f}')
        print(f'loss: {np.mean(orig_loss_prime):.4f}  {np.mean(unlearn_loss_prime):.4f}  {np.mean(original_loss):.4f}')
        print('-------------------------------------------------')
    return


# def adv_unlearn(args, model, data, edges_to_forget, device):
#     parameters = [p for p in model.parameters() if p.requires_grad]

#     # cg_loss = 0.
#     # status_count = defaultdict(int)
#     # pbar = tqdm(edges_to_forget, desc=f'unlearning {min_num_edges} edges', total=min_num_edges)
#     for edge in tqdm(edges_to_forget):
#         # inverse_hvps, status = influence(args, data, edge, device=device)
#         inverse_hvps, _, _ = _influence_new(args, data, edge, device=device)
#         with torch.no_grad():
#             delta = [p + infl for p, infl in zip(parameters, inverse_hvps)]
#             # delta = [p + infl / data['num_nodes'] for p, infl in zip(parameters, inverse_hvps)]
#             for i, p in enumerate(parameters):
#                 p.copy_(delta[i])
#     return model

def adv_batch_unlearn(args, model, data, edges_to_forget, device):
    parameters = [p for p in model.parameters() if p.requires_grad]
    # edges_ = [(v1, v2) for v1, v2 in data['edges'] if (v1, v2) not in edges_to_forget]
    edges_ = copy.deepcopy(data['edges'])
    for e in edges_to_forget:
        edges_.remove(e)
    # edge_index = torch.tensor(data['edges'], device=device).t()
    edge_index_prime = torch.tensor(edges_, dtype=torch.long, device=device).t()

    model.eval()

    x_train = torch.tensor(data['train_set'].nodes, device=device)
    y_train = torch.tensor(data['train_set'].labels, device=device)
    y_hat = model(x_train, edge_index_prime)
    loss_prime = model.loss_sum(y_hat, y_train)
    v = grad(loss_prime, parameters)

    # Directly approximate of H-1v
    inverse_hvps, loss, status = inverse_hvp_cg(data, model, edge_index_prime, v, args.damping, device)
    with torch.no_grad():
        delta = [p + infl for p, infl in zip(parameters, inverse_hvps)]
        # delta = [p + (infl/num_edge) for p, infl in zip(parameters, inverse_hvps)]
        # delta = [p + infl / data['num_nodes'] for p, infl in zip(parameters, inverse_hvps)]
        for i, p in enumerate(parameters):
            p.copy_(delta[i])
    return model


if __name__ == '__main__':
    data = Dataset(root='/tmp/', name='cora')
    adj, features, labels = data.adj, data.features, data.labels
    idx_train, idx_val, idx_test = data.idx_train, data.idx_val, data.idx_test
    idx_unlabeled = np.union1d(idx_val, idx_test)
    idx_unlabeled = np.union1d(idx_val, idx_test)
    print(type(adj), type(features), type(labels))
    print('adj:', adj.shape)
    print('features:', features.shape)
    print('labels:', labels.shape)

    print('idx_train:', idx_train[:10])
