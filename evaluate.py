import os
import argparse
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report
from data_loader import load_data
from utils import load_model, sample_edges


def analyze_influence_vs_loss_diff(args):
    df = pd.read_csv(os.path.join('./result', args.data, 'influence_vs_loss-diff.csv'))
    influences = df['influence'].values.astype(float)
    loss_diff = df['loss_diff'].values.astype(float)
    distance = abs(influences - loss_diff)
    num_invalide_infl = len(np.where(distance > 0.1)[0])
    print(f'The mean of distance is {np.mean(distance)}')
    print(f'The number of invalid influence is {num_invalide_infl}')


def inference_comparison(args, data, device):
    edges_to_forget = sample_edges(args, data)[:50]
    model_original = load_model(args, data)
    model_unlearned = load_model(args, data, type='unlearn')
    model_retrained = load_model(args, data, type='retrain')

    edge_index = torch.tensor(data['edges'], device=device)
    edge_index_prime = torch.tensor([e for e in data['edges'] if e not in edges_to_forget], device=device)
    test_loader = DataLoader(data['test_set'], batch_size=1024, shuffle=False)

    result = {
        'original': [],
        'unlearned': [],
        'retrained': [],
        'labels': [],
    }
    model_original.eval()
    model_unlearned.eval()
    model_retrained.eval()
    with torch.no_grad():
        for nodes, labels in test_loader:
            nodes = nodes.to(device)
            labels = nodes.to(device)
            y_hat = model_original(nodes, edge_index)
            y_pred = torch.argmax(y_hat, dim=1)
            y_hat_tilde = model_unlearned(nodes, edge_index_prime)
            y_pred_tilde = torch.argmax(y_hat_tilde, dim=1)
            y_hat_prime = model_retrained(nodes, edge_index_prime)
            y_pred_prime = torch.argmax(y_hat_prime)
            result['original'].extend(y_pred.cpu().tolist())
            result['unlearned'].extend(y_pred_tilde.cpu().tolist())
            result['retrained'].extend(y_pred_prime.cpu().tolist())
            result['labels'].extend(labels.cpu().tolist())

    for m in ['original', 'unlearned', 'retrained']:
        comparison = classification_report(result['labels'], result[m])
        print('For', m)
        print(comparison)
        print('-' * 40)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', dest='data', type=str, default='cora')
    parser.add_argument('-g', dest='gpu', type=int, default=-1)
    parser.add_argument('-l', dest='loss_diff', action='store_true',
                        help='Indicator of running analysis on influence against loss difference')
    parser.add_argument('-i', dest='inference_comparison', action='store_true',
                        help='Indicator of evaluating the unlearning model by inference comparison.')
    args = parser.parse_args()

    data = load_data(args.data)
    device = torch.device(f'cuda:{args.gpu}') if torch.cuda.is_available() and args.gpu >= 0 else torch.device('cpu')

    if args.loss_diff:
        analyze_influence_vs_loss_diff(args)

    if args.inference_comparison:
        inference_comparison(args, data, device)
