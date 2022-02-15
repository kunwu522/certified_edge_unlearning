import numpy as np
import torch
from torch.autograd import grad
from tqdm import tqdm
from utils import load_model, sample_nodes, loss_of_test_nodes


def hessian_vector_product(model, edge_index, x, y, v, device):
    y_hat = model(x, edge_index)
    train_loss = model.loss(y_hat.unsqueeze(0), y.unsqueeze(0))
    parameters = [p for p in model.parameters() if p.requires_grad]
    grads = grad(train_loss, parameters, retain_graph=True, create_graph=True)
    vps = [torch.sum(g * v_elem, dim=-1) for g, v_elem in zip(grads, v)]

    hvps = []
    for vp, parameter in zip(vps, parameters):
        if vp.dim() == 0:
            hvp = grad(vp, parameter, retain_graph=True)[0]
        elif vp.dim() == 1:
            vp = vp[vp != 0]
            if vp.nelement() == 0:
                hvp = torch.zeros_like(parameter, device=device)
            else:
                hvp = torch.sum(torch.cat([grad(_vp, parameter, retain_graph=True)
                                [0].unsqueeze(0) for _vp in vp], dim=0), dim=0)
        hvps.append(hvp)

    return hvps


def influence(args, data, edge_to_forget, test_node=None, device=torch.device('cpu')):
    model = load_model(args, data).to(device)

    r = args.r
    recursion_depth = args.depth
    scale = args.scale

    edges_ = [(v1, v2) for v1, v2 in data['edges']]
    edges_.remove(edge_to_forget)
    edge_index_prime = torch.tensor(edges_, dtype=torch.long, device=device).t()

    edge_index = torch.tensor(data['edges'], device=device).t()

    parameters = [p for p in model.parameters() if p.requires_grad]

    model.eval()

    x_train = torch.tensor(data['train_set'].nodes, device=device)
    y_train = torch.tensor(data['train_set'].labels, device=device)
    y_hat = model(x_train, edge_index_prime)
    loss_prime = model.loss_sum(y_hat, y_train)

    y_hat = model(x_train, edge_index)
    loss = model.loss_sum(y_hat, y_train)
    delta_e = loss_prime - loss
    grad_delta_e = grad(delta_e, parameters)

    if test_node is not None:
        y_hat = model(torch.tensor(test_node['node'], device=device), edge_index_prime).unsqueeze(0)
        loss_test = model.loss(y_hat, torch.tensor(test_node['label'], device=device).unsqueeze(0))
        grad_loss_test = grad(loss_test, parameters)
    v = grad_delta_e if test_node is None else grad_loss_test

    inverse_hvps = None
    for _ in range(r):
        cur_estimate = v
        for t in range(recursion_depth):
            sampled_node, sampled_label = sample_nodes(data['train_set'])
            sampled_node = torch.tensor(sampled_node, device=device)
            sampled_label = torch.tensor(sampled_label, device=device)

            hvps = hessian_vector_product(model, edge_index_prime,
                                          sampled_node, sampled_label,
                                          cur_estimate, device)
            cur_estimate = [a + b - c / scale for a, b, c in zip(v, cur_estimate, hvps)]

        if inverse_hvps is None:
            inverse_hvps = [b/scale for b in cur_estimate]
        else:
            inverse_hvps = [a + b / scale for (a, b) in zip(inverse_hvps, cur_estimate)]

    inverse_hvps = [a / r for a in inverse_hvps]
    if test_node is None:
        return inverse_hvps
    else:
        return np.sum([torch.sum((inverse_hvp) * g).cpu().item() for inverse_hvp, g in zip(inverse_hvps, grad_delta_e)]) / data['num_nodes']


def unlearn(args, data, edges_to_forget, device):
    model = load_model(args, data).to(device)
    parameters = [p for p in model.parameters() if p.requires_grad]

    for edge in tqdm(edges_to_forget, desc='unlearning'):
        inverse_hvps = influence(args, data, edge, device=device)

        with torch.no_grad():
            delta = [p - (infl / data['num_nodes']) for p, infl in zip(parameters, inverse_hvps)]
            for i, p in enumerate(parameters):
                p.copy_(delta[i])

    return model


def influences(args, data, edges_to_forget, device):
    print('Start to unlearn...')
    test_node = loss_of_test_nodes(args, data, device=device)[2]
    print('test_node:', test_node)

    edges_ = [(v1, v2) for v1, v2 in data['edges']]

    results = {}
    for edge in tqdm(edges_to_forget, desc='  unlearning'):
        edges_.remove(edge)
        infl = influence(args, data, test_node, edge, device)
        results[edge] = infl
        edges_ = [(v1, v2) for v1, v2 in data['edges']]

    print('Unlearning finished.')
    print()
    return results
