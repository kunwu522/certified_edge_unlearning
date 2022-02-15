import os
import random
import torch
import argparse
import numpy as np
import pandas as pd
from data_loader import load_data
from train import train_gcn
from retrain import retrain, loss_difference
from unlearn import unlearn, influences
from utils import save_model, sample_edges


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-seed', type=int, default=522)
    parser.add_argument('-train', action='store_true', help='Indicator of training.')
    parser.add_argument('-retrain', action='store_true', help='Indicator of retraining.')
    parser.add_argument('-unlearn', action='store_true', help='Indicator of unlearning.')
    parser.add_argument('-influence', action='store_true',
                        help='Indicator of running a experiment on "influence vs loss difference".')

    parser.add_argument('-save', action='store_true', help='save the result to a file.')

    # For training
    parser.add_argument('-g', dest='gpu', type=int, default=-1)
    parser.add_argument('-d', dest='data', type=str, default='cora')
    parser.add_argument('-hidden', type=int, default=32)
    parser.add_argument('-epochs', type=int, default=100)
    parser.add_argument('-batch', type=int, default=512)
    parser.add_argument('-test-batch', type=int, default=1024)
    parser.add_argument('-lr', type=float, default=0.001)
    parser.add_argument('-l2', type=float, default=1E-5)
    parser.add_argument('-emb-dim', type=int, default=32)
    parser.add_argument('-p', dest='patience', type=int, default=10)

    # For unlearning
    parser.add_argument('-edge-path', dest='edges', type=str, default=None)
    parser.add_argument('-depth', type=int, default=1000)
    parser.add_argument('-r', type=int, default=10)
    parser.add_argument('-scale', type=int, default=1)
    parser.add_argument('-num_edges', type=int, default=50, 
                            help='subsample the 10% edges for speed up the experiments')
    args = parser.parse_args()

    print('Parameters:', vars(args))

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    data = load_data(args.data)
    device = torch.device(f'cuda:{args.gpu}') if torch.cuda.is_available() and args.gpu >= 0 else torch.device('cpu')

    if args.train:
        model = train_gcn(args, data, device=device)
        torch.save(model.state_dict(), os.path.join('./checkpoint', f'gcn_{args.data}_best.pt'))

    if args.retrain:
        edges_to_forget = sample_edges(args, data)
        retrain(args, data, edges_to_forget, device)

    if args.influence:
        if args.num_edges != -1:
            edges_to_forget = sample_edges(args, data)[:args.num_edges]
        else:
            edges_to_forget = sample_edges(args, data)

        loss_diff = loss_difference(args, data, edges_to_forget, device)
        infls = influences(args, data, edges_to_forget, device)
        if args.save:
            df = pd.DataFrame({
                'influence': infls.values(),
                'loss_diff': loss_diff.values(),
            }, index=edges_to_forget)

            df.to_csv(os.path.join('./result', args.data, 'influence_vs_loss-diff.csv'))

        for e in edges_to_forget:
            print(f'Edge {e}, influence: {infls[e]:.4f}, loss_diff: {loss_diff[e]:.4f}.')

    if args.unlearn:
        if args.num_edges != -1:
            edges_to_forget = sample_edges(args, data)[:args.num_edges]
        else:
            edges_to_forget = sample_edges(args, data)

        model_unlearned = unlearn(args, data, edges_to_forget, device)
        save_model(args, model_unlearned, type='unlearn')
        model_retrained = retrain(args, data, edges_to_forget, device, forget_all=True)
        save_model(args, model_retrained, type='retrain')


        