import os
import copy
import random
import time
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report
from argument import get_args
from data_loader import load_data
from mia import MIA_accuracy
from train import evaluate
from retrain import retrain
from hessian import hessian
from utils import load_model, sample_edges, model_path


def performance(args, data, device):
    # print(model_path(args, args.performance_type, args.edges[0]))
    model = load_model(args, data, type=args.performance_type, edges=args.edges[0]).to(device)
    if args.performance_type == 'retrain' or args.performance_type == 'unlearn':
        _edges = sample_edges(args, data, method=args.method)[:args.edges[0]]
        print('!!!', _edges[-20:])
        data['edges'] = [e for e in data['edges'] if e not in _edges]
    evaluate(args, data, model, device)


def analyze_influence_vs_loss_diff(args):
    df = pd.read_csv(os.path.join('./result', args.data, 'influence_vs_loss-diff.csv'))
    influences = df['influence'].values.astype(float)
    loss_diff = df['loss_diff'].values.astype(float)
    distance = abs(influences - loss_diff)
    num_invalide_infl = len(np.where(distance > 0.1)[0])
    print(f'The mean of distance is {np.mean(distance)}')
    print(f'The number of invalid influence is {num_invalide_infl}')


def inference_comparison(args, data, device):
    edges_to_forget = sample_edges(args, data, args.method)
    edge_index = torch.tensor(data['edges'], device=device).t()
    test_loader = DataLoader(data['test_set'], batch_size=1024, shuffle=False)

    y_preds = []
    y_trues = []
    original_loss = []
    model_original = load_model(args, data).to(device)
    model_original.eval()
    with torch.no_grad():
        for nodes, labels in test_loader:
            nodes, labels = nodes.to(device), labels.to(device)
            y_hat = model_original(nodes, edge_index)
            original_loss.append(model_original.loss(y_hat, labels).cpu().item())
            y_pred = torch.argmax(y_hat, dim=1)
            y_preds.extend(y_pred.cpu().tolist())
            y_trues.extend(labels.cpu().tolist())
    del model_original
    torch.cuda.empty_cache()
    res_original = classification_report(y_trues, y_preds, digits=4, output_dict=True)

    print('Original: ')
    print(f'          {res_original["accuracy"]:.4f} & {res_original["macro avg"]["precision"]:.4f} & {res_original["macro avg"]["precision"]:.4f} & {res_original["macro avg"]["f1-score"]:.4f} & {np.mean(original_loss):.4f}')
    print('=================================================')
    for num_edges in args.edges:
        edge_index_prime = torch.tensor(
            [e for e in data['edges'] if e not in edges_to_forget[:num_edges]], device=device).t()

        y_preds, y_trues = [], []
        unlearn_loss = []
        model_unlearned = load_model(args, data, type='unlearn', edges=num_edges).to(device)
        model_unlearned.eval()
        with torch.no_grad():
            for nodes, labels in test_loader:
                nodes, labels = nodes.to(device), labels.to(device)
                y_hat = model_unlearned(nodes, edge_index_prime)
                unlearn_loss.append(model_unlearned.loss(y_hat, labels).cpu().item())
                y_pred = torch.argmax(y_hat, dim=1)
                y_preds.extend(y_pred.cpu().tolist())
                y_trues.extend(labels.cpu().tolist())
        del model_unlearned
        torch.cuda.empty_cache()
        res_unlearn = classification_report(y_trues, y_preds, digits=4, output_dict=True)

        y_preds, y_trues = [], []
        retrain_loss = []
        model_retrained = load_model(args, data, type='retrain', edges=num_edges).to(device)
        model_retrained.eval()
        with torch.no_grad():
            for nodes, labels in test_loader:
                nodes = nodes.to(device)
                labels = labels.to(device)
                y_hat = model_retrained(nodes, edge_index_prime)
                retrain_loss.append(model_retrained.loss(y_hat, labels).cpu().item())
                y_pred = torch.argmax(y_hat, dim=1)
                y_preds.extend(y_pred.cpu().tolist())
                y_trues.extend(labels.cpu().tolist())
        del model_retrained
        torch.cuda.empty_cache()
        res_retrain = classification_report(y_trues, y_preds, digits=4, output_dict=True)
        print(f'                 {num_edges}                            ')
        print(
            f'Retrain:  {res_retrain["accuracy"]:.4f} {res_retrain["macro avg"]["precision"]:.4f} {res_retrain["macro avg"]["recall"]:.4f} {res_retrain["macro avg"]["f1-score"]:.4f} {np.mean(retrain_loss):.4f}')
        print(
            f'Unlearn:  {res_unlearn["accuracy"]:.4f} {res_retrain["macro avg"]["precision"]:.4f} {res_retrain["macro avg"]["recall"]:.4f} {res_unlearn["macro avg"]["f1-score"]:.4f} {np.mean(unlearn_loss):.4f} {abs(res_retrain["accuracy"] - res_unlearn["accuracy"]):.4f}')
        print(f'Diff.: {abs(res_retrain["accuracy"] - res_unlearn["accuracy"]):.4f}')
        print(f'loss: {np.mean(unlearn_loss):.4f}  {np.mean(retrain_loss):.4f}')
        print('-------------------------------------------------')


def l2_distance(args, data):
    # edges_to_forget = sample_edges(args, data, args.method, args.edges)
    model_original = load_model(args, data)
    model_unlearned = load_model(args, data, type='unlearn')
    model_retrained = load_model(args, data, type='retrain')
    result = {'A&A_tilde': [], 'A&A_prime': [], 'A_prime&A_tilde': []}
    for p_original, p_unlearned, p_retrained in zip(model_original.parameters(), model_unlearned.parameters(), model_retrained.parameters()):
        result['A&A_tilde'].append(np.linalg.norm(p_original.detach() - p_unlearned.detach()))
        result['A&A_prime'].append(np.linalg.norm(p_original.detach() - p_retrained.detach()))
        result['A_prime&A_tilde'].append(np.linalg.norm(p_retrained.detach() - p_unlearned.detach()))

    # for _, v in result.items():
    #     v = np.sum(v)
    print('L2 Distance:')
    print('  ', result)


def mia_attack(args, data, device):
    num_edges_to_forget = args.edges[0]
    # get edges that need to forget
    edges_to_forget = sample_edges(args, data, args.method)[:num_edges_to_forget]
    edges_ = [e for e in data['edges'] if e not in edges_to_forget]

    member_edges = random.sample([e for e in data['edges'] if e not in edges_to_forget], int(0.5 * len(data['edges'])))
    non_member_edges = []
    while len(non_member_edges) < len(member_edges):
        v1 = random.randint(0, data['num_nodes']-1)
        v2 = random.randint(0, data['num_nodes']-1)
        if v1 == v2:
            continue
        if (v1, v2) in data['edges']:
            continue
        non_member_edges.append((v1, v2))

    mia_edges = member_edges + non_member_edges
    mia_labels = [1] * len(member_edges) + [0] * len(non_member_edges)

    # shuffle the edges
    random_indices = list(range(len(mia_edges)))
    random.shuffle(random_indices)
    mia_edges = np.array(mia_edges)[random_indices]
    mia_labels = np.array(mia_labels)[random_indices]

    nodes = torch.tensor(data['nodes'], device=device)
    edge_index = torch.tensor(data['edges'], device=device).t()
    edge_index_prime = torch.tensor(edges_, device=device).t()
    result = {'original': [], 'unlearn': [], 'retrain': []}
    for _ in range(10):
        for m in ['original', 'unlearn', 'retrain']:
            model = load_model(args, data, type=m, edges=num_edges_to_forget).to(device)
            model.eval()
            with torch.no_grad():
                if m == 'original':
                    y_hat = model(nodes, edge_index)
                else:
                    y_hat = model(nodes, edge_index_prime)
                posterior = F.softmax(y_hat, dim=1)
            acc = MIA_accuracy(mia_edges, mia_labels,
                               np.array(edges_to_forget), np.ones(len(edges_to_forget)
                                                                  ) if m == 'original' else np.zeros(len(edges_to_forget)),
                               data['features'], posterior.cpu().numpy())
            result[m].append(acc)
    for m, acc in result.items():
        print(f'For {m}, the MIA accuracy is {np.mean(acc):.5f}.')


def test_mia_with_diff_nodes(args, data, device):
    disjoin_edges = []
    for v1, v2 in data['edges']:
        label1, label2 = data['labels'][v1], data['labels'][v2]
        if label1 != label2:
            disjoin_edges.append((v1, v2))

    results = {'original': [], 'retrain': []}
    for _ in range(10):
        # edges_to_forget = random.sample(disjoin_edges, int(args.edges * 0.01 * data['num_edges']))
        edges_to_forget = sample_edges(args, data, args.method, args.edges)
        member_edges = random.sample([e for e in data['edges'] if e not in edges_to_forget],
                                     int(0.7 * len(data['edges'])))
        non_member_edges = []
        while len(non_member_edges) < len(member_edges):
            v1 = random.randint(0, data['num_nodes']-1)
            v2 = random.randint(0, data['num_nodes']-1)
            if v1 == v2:
                continue
            if (v1, v2) in data['edges']:
                continue
            non_member_edges.append((v1, v2))

        mia_edges = member_edges + non_member_edges
        mia_labels = [1] * len(member_edges) + [0] * len(non_member_edges)

        # shuffle the edges
        random_indices = list(range(len(mia_edges)))
        random.shuffle(random_indices)
        mia_edges = np.array(mia_edges)[random_indices]
        mia_labels = np.array(mia_labels)[random_indices]

        nodes = torch.tensor(data['nodes'], device=device)
        edge_index = torch.tensor(data['edges'], device=device).t()
        retrain_model = retrain(args, data, edges_to_forget, device, forget_all=True)
        retrain_model.eval()
        with torch.no_grad():
            y_hat = retrain_model(nodes, edge_index)
            posterior = F.softmax(y_hat, dim=1)
        retrain_acc = MIA_accuracy(mia_edges, mia_labels, np.array(edges_to_forget), np.zeros(
            len(edges_to_forget)), data['features'], posterior.cpu().numpy())
        results['retrain'].append(retrain_acc)

        original_model = load_model(args, data).to(device)
        original_model.eval()
        with torch.no_grad():
            y_hat = original_model(nodes, edge_index)
            posterior = F.softmax(y_hat, dim=1)
        original_acc = MIA_accuracy(mia_edges, mia_labels, np.array(edges_to_forget), np.ones(
            len(edges_to_forget)), data['features'], posterior.cpu().numpy(), True)

        results['original'].append(original_acc)
    print(f'MIA accuracy on original model with {args.edges}% edges: {np.mean(results["original"]):.4f}.')
    print(f'MIA accuracy on retrained model with {args.edges}% edges: {np.mean(results["retrain"]):.4f}.')


# def condition_number(A, tol=1e-5):
#     E = torch.linalg.eigvalsh(A)
#     return torch.abs(torch.max(E)) / torch.abs(torch.min(E))


def _is_PD(A):
    try:
        _ = torch.linalg.cholesky(A)
    except RuntimeError as e:
        return False

    return True


eigenvalue_threshold = {
    'gcn': [0, 1, 9500],
    # 'gcn': [1, 9500],
    'gat': [0, 1, 1, 1, 180],
    'gin': [0, 480, 2, 80, 1],
    'sage': [0, 150, 1, 150],
}


def condition_number(args, data, device):
    model = load_model(args, data, 'original').to(device)

    edge_index = torch.tensor(data['edges'], device=device).t()
    x_train = torch.tensor(data['train_set'].nodes, device=device)
    y_train = torch.tensor(data['train_set'].labels, device=device)

    model.eval()
    t0 = time.time()
    print(f'Start to calculate Hessian matrix of {args.model}')
    H = hessian(model, edge_index, x_train, y_train)
    print(f'Calculate hessian on G, duration {(time.time()-t0):.4f}.')

    print('------------------------------------')
    print('Model:', args.model)
    print('Dataset:', args.data)
    print(f'Setting, layers:{1 if len(args.hidden) == 0 else len(args.hidden) + 1}')
    for idx, h in enumerate(H):
        t0 = time.time()
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
        E = torch.linalg.eigvalsh(h.view(n, -1))
        print('E', E)
        plt.figure()
        plt.plot(np.arange(len(E)), E.numpy())
        plt.savefig(f'./eigenvalues_{args.model}_{args.data}_{idx}.png')

        E = E[eigenvalue_threshold[args.model][idx]:]
        # print(E)
        print('condition number:', (torch.max(E) / torch.min(E)).item())
        print('max eigenvalue:', torch.max(E).item())
        # t0 = time.time()
        # is_convex, max_eig_v = _is_PSD(h.view(n, -1))
        # print('  is convex:', is_convex)
        # print('  max eig value:', max_eig_v)

        print(f'  duration: {(time.time() - t0):.2}s.')


if __name__ == '__main__':
    args = get_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    data = load_data(args)
    device = torch.device(f'cuda:{args.gpu}') if torch.cuda.is_available() and args.gpu >= 0 else torch.device('cpu')

    if args.test_mia:
        test_mia_with_diff_nodes(args, data, device)

    if args.loss_diff:
        analyze_influence_vs_loss_diff(args)

    if args.inference_comparison:
        inference_comparison(args, data, device)

    if args.l2_distance:
        l2_distance(args, data)

    if args.mia_attack:
        mia_attack(args, data, device)

    if args.performance:
        performance(args, data, device)

    if args.condition_number:
        condition_number(args, data, device)
