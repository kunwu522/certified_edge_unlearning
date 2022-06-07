import os
import time
import torch
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader

from utils import create_model
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

        if args.early_stop:
            if valid_loss < best_valid_loss:
                best_valid_loss = valid_loss
                trail_count = 0
                best_epoch = e
                torch.save(model.state_dict(), os.path.join('./checkpoint',
                                                            'tmp', f'{args.model}_{args.data}_{args.gpu}_best.pt'))
            else:
                trail_count += 1
                if trail_count > patience:
                    if verbose:
                        print(f'  Early Stop, the best Epoch is {best_epoch}, validation loss: {best_valid_loss:.4f}.')
                    break
        else:
            torch.save(model.state_dict(), os.path.join('./checkpoint',
                                                        'tmp', f'{args.model}_{args.data}_{args.gpu}_best.pt'))

    return best_epoch


def evaluate(args, data, model, device):
    test_loader = DataLoader(data['test_set'], batch_size=args.test_batch, shuffle=False)
    edge_index = torch.tensor(data['edges'], device=device).t()
    y_preds = []
    y_true = []

    model.eval()
    with torch.no_grad():
        for nodes, labels in test_loader:
            nodes = nodes.to(device)
            labels = labels.to(device)

            y_hat = model(nodes, edge_index)
            y_pred = torch.argmax(y_hat, dim=1)

            y_preds.extend(y_pred.cpu().tolist())
            y_true.extend(labels.cpu().tolist())

    results = classification_report(y_true, y_preds, digits=4)
    print('  Result:')
    print(results)


def test(model, test_loader, edge_index, device):
    y_preds = []
    y_trues = []
    test_loss = []
    model.eval()
    with torch.no_grad():
        for nodes, labels in test_loader:
            nodes, labels = nodes.to(device), labels.to(device)
            y_hat = model(nodes, edge_index)
            test_loss.append(model.loss(y_hat, labels).cpu().item())
            y_pred = torch.argmax(y_hat, dim=1)
            y_preds.extend(y_pred.cpu().tolist())
            y_trues.extend(labels.cpu().tolist())
    # del model
    # torch.cuda.empty_cache()
    res = classification_report(y_trues, y_preds, digits=4, output_dict=True)
    return res, np.mean(test_loss)


def train_model(args, data, eval=True, verbose=True, device=torch.device('cpu'), return_epoch=False):
    if verbose:
        t0 = time.time()
        print(f'Start to train a {args.model} model...')

    model = create_model(args, data).to(device)
    num_epochs = train(args, data, model, device, verbose)
    model.load_state_dict(torch.load(os.path.join('./checkpoint', 'tmp',
                          f'{args.model}_{args.data}_{args.gpu}_best.pt')))

    if eval:
        evaluate(args, data, model, device)

    os.remove(os.path.join('./checkpoint', 'tmp', f'{args.model}_{args.data}_{args.gpu}_best.pt'))

    if verbose:
        print(f'{args.model} model training finished. Duration:', int(time.time() - t0))

    if return_epoch:
        return model, num_epochs
    else:
        return model
