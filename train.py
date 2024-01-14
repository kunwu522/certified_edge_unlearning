import os
import time
import torch
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader

from utils import create_model
from sklearn.metrics import classification_report


def train(args, data, model, device, verbose):
    train_loader = DataLoader(data.train_set, batch_size=args.batch, shuffle=True)
    valid_loader = DataLoader(data.valid_set, batch_size=args.test_batch)
    if args.transductive_edge:
        train_edge_index = torch.tensor(data.edges, device=device).t()
        valid_edge_index = torch.tensor(data.edges, device=device).t()
    else:
        train_edge_index = torch.tensor(data.train_edges, device=device).t()
        valid_edge_index = torch.tensor(data.valid_edges, device=device).t()

    num_epochs = args.epochs
    lr = args.lr
    l2 = args.l2
    patience = args.patience

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=l2)

    best_valid_loss = 999999.
    best_epoch = 0
    trail_count = 0
    best_model_state = model.state_dict()

    for e in range(1, num_epochs + 1):
        train_losses = []
        model.train()
        iterator = tqdm(train_loader, f'  Epoch {e}') if verbose else train_loader
        for nodes, labels in iterator:
            model.zero_grad()
            nodes = nodes.to(device)
            labels = labels.to(device)

            y_hat = model(nodes, train_edge_index)
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

                y_hat = model(nodes, valid_edge_index)
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
                # torch.save(model.state_dict(), os.path.join('./checkpoint',
                #                                             'tmp', f'{args.model}_{args.data}_{args.gpu}_best.pt'))
                best_model_state = model.state_dict()
            else:
                trail_count += 1
                if trail_count > patience:
                    if verbose:
                        print(f'  Early Stop, the best Epoch is {best_epoch}, validation loss: {best_valid_loss:.4f}.')
                    break
        else:
            # torch.save(model.state_dict(), os.path.join('./checkpoint',
            #                                             'tmp', f'{args.model}_{args.data}_{args.gpu}_best.pt'))
            best_model_state = model.state_dict()

    return best_model_state, best_epoch


def prediction(args, data, model, device):
    test_loader = DataLoader(data.test_set, batch_size=args.test_batch, shuffle=False)
    edge_index = torch.tensor(data.edges, device=device).t()
    # train_loader = DataLoader(data.train_set, batch_size=args.test_batch, shuffle=False)
    # edge_index = torch.tensor(data.train_edges, device=device).t()

    y_preds = []
    model.eval()
    with torch.no_grad():
        for nodes, _ in test_loader:
            nodes = nodes.to(device)
            y_hat = model(nodes, edge_index)
            y_pred = torch.argmax(y_hat, dim=1)
            y_preds.append(y_pred.cpu().numpy())

    return np.concatenate(y_preds, axis=0)


def evaluate(args, data, model, device):
    train_loader = DataLoader(data.train_set, batch_size=args.batch, shuffle=False)
    valid_loader = DataLoader(data.valid_set, batch_size=args.test_batch, shuffle=False)
    test_loader = DataLoader(data.test_set, batch_size=args.test_batch, shuffle=False)
    if args.transductive_edge:
        train_edge_index = torch.tensor(data.edges, device=device).t()
        valid_edge_index = torch.tensor(data.edges, device=device).t()
    else:
        train_edge_index = torch.tensor(data.train_edges, device=device).t()
        valid_edge_index = torch.tensor(data.valid_edges, device=device).t()
    # train_edge_index = torch.tensor(data.edges, device=device).t()
    # valid_edge_index = torch.tensor(data.edges, device=device).t()
    edge_index = torch.tensor(data.edges, device=device).t()
    y_preds = []
    y_true = []
    train_loss, valid_loss, test_loss = [], [], []

    model.eval()
    with torch.no_grad():
        for nodes, labels in train_loader:
            nodes = nodes.to(device)
            labels = labels.to(device)
            y_hat = model(nodes, train_edge_index)
            loss = model.loss(y_hat, labels).cpu().item()
            train_loss.append(loss)

        for nodes, labels in valid_loader:
            nodes = nodes.to(device)
            labels = labels.to(device)
            y_hat = model(nodes, valid_edge_index)
            loss = model.loss(y_hat, labels).cpu().item()
            valid_loss.append(loss)

        for nodes, labels in test_loader:
            nodes = nodes.to(device)
            labels = labels.to(device)
            y_hat = model(nodes, edge_index)
            loss = model.loss(y_hat, labels).cpu().item()
            test_loss.append(loss)
            y_pred = torch.argmax(y_hat, dim=1)
            # print('y_pred', y_pred)
            y_preds.extend(y_pred.cpu().tolist())
            y_true.extend(labels.cpu().tolist())

    # print('  ', np.unique(y_preds, return_counts=True))
    results = classification_report(y_true, y_preds, digits=4, output_dict=True)
    return {
        'train_loss': np.mean(train_loss),
        'valid_loss': np.mean(valid_loss),
        'test_loss': np.mean(test_loss),
        'accuracy': results['accuracy'],
        'percision': results['macro avg']['precision'],
        'recall': results['macro avg']['recall'],
        'f1': results['macro avg']['f1-score'],
        'predictions': y_preds,
    }


# def test(args, model, test_loader, edge_index, device):
    # y_preds = []
    # y_trues = []
    # test_loss = []
    # model.eval()
    # with torch.no_grad():
    #     for nodes, labels in test_loader:
    #         nodes, labels = nodes.to(device), labels.to(device)
    #         y_hat = model(nodes, edge_index)
    #         if args.log:
    #             loss = model.log_ce_loss(y_hat, labels).cpu().item()
    #         elif args.hinge:
    #             loss = model.hinge_loss(y_hat, labels).cpu().item()
    #         else:
    #             loss = model.loss(y_hat, labels).cpu().item()
    #         test_loss.append(loss)
    #         y_pred = torch.argmax(y_hat, dim=1)
    #         y_preds.extend(y_pred.cpu().tolist())
    #         y_trues.extend(labels.cpu().tolist())

    # res = classification_report(y_trues, y_preds, digits=4, output_dict=True)
    # return res, np.mean(test_loss)


def train_model(args, data, eval=True, verbose=True, device=torch.device('cpu'), return_epoch=False):
    if verbose:
        t0 = time.time()
        print(f'Start to train a {args.model} model...')

    model = create_model(args, data).to(device)
    model_state, num_epochs = train(args, data, model, device, verbose)
    model.load_state_dict(model_state)
    if eval:
        evaluate(args, data, model, device)

    if verbose:
        print(f'{args.model} model training finished. Duration:', int(time.time() - t0))

    if return_epoch:
        return model, num_epochs
    else:
        return model
