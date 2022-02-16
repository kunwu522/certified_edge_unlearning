import os
import torch
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader

from model import GCN
from sklearn.metrics import classification_report


def train(args, data, model, device, verbose):
    train_loader = DataLoader(data['train_set'], batch_size=args.batch, shuffle=True)
    valid_loader = DataLoader(data['valid_set'], batch_size=args.test_batch)
    edge_index = torch.tensor(data['edges'], device=device).t()

    num_epochs = args.epochs
    lr = args.lr
    l2 = args.l2
    patience = args.patience

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=l2)

    best_valid_loss = 999999.
    best_epoch = 0
    trail_count = 0

    for e in range(1, num_epochs + 1):

        train_losses = []

        model.train()
        iterator = tqdm(train_loader, f'  Epoch {e}') if verbose else train_loader
        for nodes, labels in iterator:
            model.zero_grad()

            nodes = nodes.to(device)
            labels = labels.to(device)

            y_hat = model(nodes, edge_index)
            loss = model.loss(y_hat, labels)
            loss.backward()
            optimizer.step()

            train_losses.append(loss.cpu().item())

        train_loss = np.mean(train_losses)

        valid_losses = []

        model.eval()
        with torch.no_grad():
            for nodes, labels in valid_loader:
                nodes = nodes.to(device)
                labels = labels.to(device)

                y_hat = model(nodes, edge_index)
                loss = model.loss(y_hat, labels)

                valid_losses.append(loss.cpu().item())

        valid_loss = np.mean(valid_losses)

        if verbose:
            print(f'  Epoch {e}, training loss: {train_loss:.4f}, validation loss: {valid_loss:.4f}.')

        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            trail_count = 0
            best_epoch = e
            torch.save(model.state_dict(), os.path.join('./checkpoint', f'tmp_gcn_{args.data}_{args.gpu}_best.pt'))
        else:
            trail_count += 1
            if trail_count == patience:
                print(f'  Early Stop, the best Epoch is {best_epoch}, validation loss: {best_valid_loss:.4f}.')
                break


def evaluate(args, data, model, device):
    test_loader = DataLoader(data['test_set'], batch_size=args.test_batch)
    edge_index = torch.tensor(data['edges'], device=device).t()
    y_preds = []
    y_true = []

    print(type(model))
    model.eval()
    with torch.no_grad():
        for nodes, labels in test_loader:
            nodes = nodes.to(device)
            labels = labels.to(device)

            y_hat = model(nodes, edge_index)
            y_pred = torch.argmax(y_hat, dim=1)

            y_preds.extend(y_pred.cpu().tolist())
            y_true.extend(labels.cpu().tolist())

    results = classification_report(y_true, y_preds)
    print('  Result:')
    print(results)


def train_gcn(args, data, eval=True, verbose=True, device=torch.device('cpu')):
    if verbose:
        print('Start to train a GCN model...')

    embedding_size = args.emb_dim if data['features'] is None else data['features'].shape[1]
    model = GCN(data['num_nodes'], embedding_size,
                args.hidden, data['num_classes'], data['features']).to(device)
    train(args, data, model, device, verbose)
    model.load_state_dict(torch.load(os.path.join('./checkpoint', f'tmp_gcn_{args.data}_{args.gpu}_best.pt')))

    if eval:
        evaluate(args, data, model, device)

    os.remove(os.path.join('./checkpoint', f'tmp_gcn_{args.data}_{args.gpu}_best.pt'))

    if verbose:
        print('GCN model training finished.')

    return model
