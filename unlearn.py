from functools import reduce
import math
import numpy as np
import time
import torch
import copy
import random
import torch.nn.functional as F
from collections import defaultdict
from torch.autograd import grad
from scipy.optimize import fmin_ncg, fmin_l_bfgs_b, fmin_cg
from hessian import hessian_vector_product

import cgu.utils as cgu_utils
from train import evaluate
from mia import evaluate_mia_model
from lib.lissa import s_infected_nodes, hvp
from lib.torch_influence import CGPertuabtionInfluence, EdgeInfluence


def _hessian_vector_product(model, edge_index, x, y, v, device):
    y_hat = model(x, edge_index)
    train_loss = model.loss_sum(y_hat, y)
    parameters = [p for p in model.parameters() if p.requires_grad]

    with torch.enable_grad():
        grads = grad(train_loss, parameters, create_graph=True)
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


def to_vector(v):
    if isinstance(v, tuple) or isinstance(v, list):
        # return v.cpu().numpy().reshape(-1)
        return np.concatenate([vv.cpu().numpy().reshape(-1) for vv in v])
    else:
        return v.cpu().numpy().reshape(-1)


def to_list(v, sizes, device):
    _v = v
    result = []
    for size in sizes:
        total = reduce(lambda a, b: a * b, size)
        result.append(_v[:total].reshape(size).float().to(device))
        _v = _v[total:]
    return tuple(result)


def _mini_batch_hvp(x, **kwargs):
    model = kwargs['model']
    x_train = kwargs['x_train']
    y_train = kwargs['y_train']
    edge_index = kwargs['edge_index']
    damping = kwargs['damping']
    device = kwargs['device']
    sizes = kwargs['sizes']
    p_idx = kwargs['p_idx']
    use_torch = kwargs['use_torch']

    x = to_list(x, sizes, device)
    if use_torch:
        _hvp = hessian_vector_product(model, edge_index, x_train, y_train, x, device, p_idx)
    else:
        model.eval()
        y_hat = model(x_train, edge_index)
        loss = model.loss(y_hat, y_train)
        params = [p for p in model.parameters() if p.requires_grad]
        if p_idx is not None:
            params = params[p_idx:p_idx + 1]
        _hvp = hvp(loss, params, x)
    # return _hvp[0].view(-1) + damping * x
    return [(a + damping * b).view(-1) for a, b in zip(_hvp, x)]


def _get_fmin_prime_fn(v, **kwargs):
    device = kwargs['device']

    def get_fmin_grad(x):
        # x = torch.tensor(x, dtype=torch.float, device=device)
        hvp = _mini_batch_hvp(x, **kwargs)
        return to_vector(hvp - v.view(-1))


def _get_fmin_loss_fn(v, **kwargs):
    device = kwargs['device']

    def get_fmin_loss(x):
        x = torch.tensor(x, dtype=torch.float, device=device)
        hvp = _mini_batch_hvp(x, **kwargs)
        obj = 0.5 * torch.dot(torch.cat(hvp, dim=0), x) - torch.dot(v, x)
        return obj.detach().cpu().numpy()

    return get_fmin_loss


def _get_fmin_grad_fn(v, **kwargs):
    device = kwargs['device']

    def get_fmin_grad(x):
        x = torch.tensor(x, dtype=torch.float, device=device)
        hvp = _mini_batch_hvp(x, **kwargs)
        # return to_vector(hvp - v.view(-1))
        return (torch.cat(hvp, dim=0) - v).cpu().numpy()

    return get_fmin_grad


def _get_fmin_hvp_fn(v, **kwargs):
    device = kwargs['device']

    def get_fmin_hvp(x, p):
        p = torch.tensor(p, dtype=torch.float, device=device)
        hvp = _mini_batch_hvp(p, **kwargs)
        return to_vector(hvp)
    return get_fmin_hvp


def _get_cg_callback(v, **kwargs):
    device = kwargs['device']

    def cg_callback(x):
        x = torch.tensor(x, dtype=torch.float, device=device)
        hvp = _mini_batch_hvp(x, **kwargs)
        obj = 0.5 * torch.dot(torch.cat(hvp, dim=0), x) - torch.dot(v, x)
        # obj = 0.5 * torch.dot(hvp, x) - torch.dot(v.view(-1), x)
        # g = to_vector(hvp - v.view(-1))
        g = (torch.cat(hvp, dim=0) - v).cpu().numpy()
        print(f'loss: {obj:.4f}, grad: {np.linalg.norm(g):.8f}')
    return cg_callback


def inverse_hvp_cg_hessian(Bs, vs, damping, device):
    inverse_hvp = []
    status = []
    for B, v in zip(Bs, vs):
        sizes = [v.size()]

        if v.dim() == 1:
            B = B.view(v.size(0), -1).to(device)
        elif v.dim() == 2:
            B = B.view(v.size(0) * v.size(1), -1).to(device)

        fmin_loss_fn = _get_fmin_loss_fn(v, H=B, damping=damping, device=device)
        fmin_grad_fn = _get_fmin_grad_fn(v, H=B, damping=damping, device=device)
        fmin_hvp_fn = _get_fmin_hvp_fn(v, H=B, damping=damping, device=device)

        res = fmin_ncg(
            f=fmin_loss_fn,
            x0=to_vector(v),
            fprime=fmin_grad_fn,
            fhess_p=fmin_hvp_fn,
            # callback=_cg_callback,
            avextol=1e-5,
            disp=False,
            full_output=True,
            maxiter=200)

        inverse_hvp.append(to_list(res[0], sizes, device)[0])
        status.append(res[5])
    return inverse_hvp, tuple(status)


def _set_attr(obj, names, val):
    if len(names) == 1:
        setattr(obj, names[0], val)
    else:
        _set_attr(getattr(obj, names[0]), names[1:], val)


def _model_params(model, with_names=True):
    # assert not self.is_model_functional
    return tuple((name, p) if with_names else p for name, p in model.named_parameters() if p.requires_grad)


def _del_attr(obj, names):
    if len(names) == 1:
        delattr(obj, names[0])
    else:
        _del_attr(getattr(obj, names[0]), names[1:])


def _model_make_functional(model, params_names):
    # assert not self.is_model_functional
    params = tuple(p.detach().requires_grad_() for p in _model_params(model, False))

    for name in params_names:
        _del_attr(model, name.split("."))

    return params


def _reshape_like_params(vec, params_shape):
    pointer = 0
    split_tensors = []
    for dim in params_shape:
        num_param = dim.numel()
        split_tensors.append(vec[pointer: pointer + num_param].view(dim))
        pointer += num_param
    return tuple(split_tensors)


def _model_reinsert_params(model, params, params_names, register=False):
    for name, p in zip(params_names, params):
        _set_attr(model, name.split("."), torch.nn.Parameter(p) if register else p)
    # self.is_model_functional = not register


def _flatten_params_like(params_like):
    vec = []
    for p in params_like:
        vec.append(p.view(-1))
    return torch.cat(vec)


def inverse_hvp(data, model, edge_index, v, damping, device):
    nodes = torch.tensor(data.train_set.nodes, device=device)
    labels = torch.tensor(data.train_set.labels, device=device)

    params_names = tuple(name for name, _ in _model_params(model))
    params_shape = tuple(p.shape for _, p in _model_params(model))
    params = _model_make_functional(model, params_names)
    flat_params = _flatten_params_like(params)
    d = flat_params.shape[0]

    def f(theta_):
        _model_reinsert_params(model, _reshape_like_params(theta_, params_shape), params_names)
        y_hat = model(nodes, edge_index)
        return model.loss(y_hat, labels)
    hess = torch.autograd.functional.hessian(f, flat_params).detach()
    with torch.no_grad():
        hess = hess + damping * torch.eye(d, device=device)
        inverse_hess = torch.inverse(hess)

    return _reshape_like_params(inverse_hess @ _flatten_params_like(v), params_shape)

def inverse_hvp_cg_sep(data, model, edge_index, vs, damping, device, use_torch=True):
    x_train = torch.tensor(data.train_set.nodes, device=device)
    y_train = torch.tensor(data.train_set.labels, device=device)
    inverse_hvp = []
    status = []
    cg_grad = []
    
    parameters = [p for p in model.parameters() if p.requires_grad]
    for i, (v, p) in enumerate(zip(vs, parameters)):
        sizes = [p.size()]
        v = v.view(-1)
        fmin_loss_fn = _get_fmin_loss_fn(v, model=model,
                                        x_train=x_train, y_train=y_train,
                                        edge_index=edge_index, damping=damping,
                                        sizes=sizes, p_idx=i, device=device,
                                        use_torch=use_torch)
        fmin_grad_fn = _get_fmin_grad_fn(v, model=model,
                                        x_train=x_train, y_train=y_train,
                                        edge_index=edge_index, damping=damping,
                                        sizes=sizes, p_idx=i, device=device,
                                        use_torch=use_torch)
        fmin_hvp_fn = _get_fmin_hvp_fn(v, model=model,
                                    x_train=x_train, y_train=y_train,
                                    edge_index=edge_index, damping=damping,
                                    sizes=sizes, p_idx=i, device=device,
                                    use_torch=use_torch)
        cg_callback = _get_cg_callback(v, model=model,
                                    x_train=x_train, y_train=y_train,
                                    edge_index=edge_index, damping=damping,
                                    sizes=sizes, p_idx=i, device=device,
                                    use_torch=use_torch)
        if use_torch:
            res = fmin_cg(
                f=fmin_loss_fn,
                x0=to_vector(v),
                fprime=fmin_grad_fn,
                gtol=1E-4,
                # norm='fro',
                # callback=cg_callback,
                disp=False,
                full_output=True,
                maxiter=100,
            )
            inverse_hvp.append(to_list(torch.from_numpy(res[0]), sizes, device)[0])
            # inverse_hvp = to_list(torch.from_numpy(res[0]), sizes, device)
            # cg_grad = np.linalg.norm(fmin_grad_fn(res[0]))
            # status = res[4]
            # print('-----------------------------------')
            # cg_grad.append(np.linalg.norm(fmin_grad_fn(res[0]), ord=np.inf))

        else:
            res = fmin_ncg(
                f=fmin_loss_fn,
                x0=to_vector(v),
                fprime=fmin_grad_fn,
                fhess_p=fmin_hvp_fn,
                # callback=cg_callback,
                avextol=1e-5,
                disp=False,
                full_output=True,
                maxiter=100)
            inverse_hvp.append(to_list(torch.from_numpy(res[0]), sizes, device)[0])
            # inverse_hvp = to_list(torch.from_numpy(res[0]), sizes, device)

        #     x, _err, d = fmin_l_bfgs_b(
        #         func=fmin_loss_fn,
        #         x0=to_vector(v),
        #         fprime=fmin_grad_fn,
        #         iprint=0,
        #     )
        #     inverse_hvp.append(to_list(x, sizes, device)[0])
        #     status.append(d['warnflag'])
        #     err += _err.item()
        # print('error:', err, status)
    return inverse_hvp, (cg_grad, status)

# def _get_fmin_fhess_fn_sgc(v, **kwargs):
#     model = kwargs['model']
#     edge_index = kwargs['edge_index']
#     w = kwargs['w']
#     nodes = kwargs['nodes']
#     labels = kwargs['labels']
#     y = F.one_hot(labels)

#     def get_fhess(x):
#         model.eval()
#         with torch.no_grad():
#             X = model.propagate(nodes, edge_index)
#             z = torch.sigmoid(y * X.mm(w.t()))
#             D = z * (1 - z)
#             H = []
#             for k in range(w.size(0)):
#                 H.append(X.t().mm(D[:, k].unsqueeze(1) * X))
#             print(torch.cat(H).view(-1, w.size(0) * X.size(1)).t().cpu().numpy().shape)
#             return torch.cat(H).view(-1, w.size(0) * X.size(1)).t().cpu().numpy()
#     return get_fhess

def _hessian_sgc(model, edge_index, w, nodes, y, lam, device):
    model.eval()
    with torch.no_grad():
        X = model.propagate(nodes, edge_index)

    with torch.no_grad():
        z = torch.sigmoid(y * X.mm(w.t()))
        D = z * (1 - z)
        H = []
        for k in range(w.size(0)):
            h = X.t().mm(D[:, k].unsqueeze(1) * X)
            h += lam * X.size(0) * torch.eye(X.size(1)).float().to(device)
            H.append(h)
        
    return torch.cat(H).to(device)

def _get_fmin_loss_fn_sgc(v, **kwargs):
    model = kwargs['model']
    edge_index = kwargs['edge_index']
    w = kwargs['w']
    lam = kwargs['lam']
    nodes = kwargs['nodes']
    labels = kwargs['labels']
    device = kwargs['device']
    y = F.one_hot(labels)

    H = _hessian_sgc(model, edge_index, w, nodes, y, lam, device)
    # print('H:', H.size())

    def get_fmin_loss(x):
        x = torch.tensor(x, dtype=torch.float, device=device)
        # print(x.size)
        hvp = H.view(w.size(0), w.size(1), -1).bmm(x.view(-1, w.size(0)).t().unsqueeze(2)).squeeze().t().flatten()
        obj = 0.5 * torch.dot(hvp, x) - torch.dot(v, x)
        return obj.detach().cpu().numpy()

    return get_fmin_loss

def _get_fmin_grad_fn_sgc(v, **kwargs):
    model = kwargs['model']
    edge_index = kwargs['edge_index']
    w = kwargs['w']
    nodes = kwargs['nodes']
    labels = kwargs['labels']
    device = kwargs['device']
    lam = kwargs['lam']
    y = F.one_hot(labels)

    H = _hessian_sgc(model, edge_index, w, nodes, y, lam, device)
    def get_fmin_grad(x):
        x = torch.tensor(x, dtype=torch.float, device=device)
        # hvp = _mini_batch_hvp(x, **kwargs)
        # hvp = H.mv(x).view(-1, w.size(0)).t()
        # print(H.view(w.size(0), w.size(1), -1).bmm(x.view(-1, w.size(0)).t().unsqueeze(2)).size())
        hvp = H.view(w.size(0), w.size(1), -1).bmm(x.view(-1, w.size(0)).t().unsqueeze(2)).squeeze().t().flatten()
        # return to_vector(hvp - v.view(-1))
        return (hvp - v).cpu().numpy()

    return get_fmin_grad


def _get_fmin_hvp_fn_sgc(v, **kwargs):
    model = kwargs['model']
    edge_index = kwargs['edge_index']
    w = kwargs['w']
    nodes = kwargs['nodes']
    labels = kwargs['labels']
    device = kwargs['device']
    lam = kwargs['lam']
    y = F.one_hot(labels)

    H = _hessian_sgc(model, edge_index, w, nodes, y, lam, device)
    def get_fmin_hvp(x, p):
        p = torch.tensor(p, dtype=torch.float, device=device)
        # hvp = _mini_batch_hvp(p, **kwargs)
        with torch.no_grad():
            hvp = H.view(w.size(0), w.size(1), -1).bmm(p.view(-1, w.size(0)).t().unsqueeze(2)).squeeze().t().flatten()
        return hvp.cpu().numpy()
    return get_fmin_hvp


def inverse_hvp_cg_sgc(data, model, edge_index, vs, lam, device):
    w = [p for p in model.parameters() if p.requires_grad][0]
    x_train = torch.tensor(data.train_set.nodes, device=device)
    y_train = torch.tensor(data.train_set.labels, device=device)
    inverse_hvp = []
    status = []
    cg_grad = []
    sizes = [p.size() for p in model.parameters() if p.requires_grad]
    v = torch.cat([vv.view(-1) for vv in vs])
    i = None
    fmin_loss_fn = _get_fmin_loss_fn_sgc(v, model=model, w=w, lam=lam,
                                         nodes=x_train, labels=y_train,
                                         edge_index=edge_index, device=device)
    fmin_grad_fn = _get_fmin_grad_fn_sgc(v, model=model, w=w, lam=lam,
                                         nodes=x_train, labels=y_train,
                                         edge_index=edge_index, device=device)
    fmin_hvp_fn = _get_fmin_hvp_fn_sgc(v, model=model, w=w, lam=lam,
                                           nodes=x_train, labels=y_train,
                                           edge_index=edge_index, device=device)
    
    res = fmin_ncg(
        f=fmin_loss_fn,
        x0=to_vector(vs),
        fprime=fmin_grad_fn,
        fhess_p=fmin_hvp_fn,
        # fhess=fmin_fhess_fn,
        # callback=cg_callback,
        avextol=1e-5,
        disp=False,
        full_output=True,
        maxiter=100)
    # inverse_hvp.append(to_list(res[0], sizes, device)[0])
    inverse_hvp = to_list(torch.from_numpy(res[0]), sizes, device)
    # print('-----------------------------------')
    status = res[5]
    cg_grad = np.linalg.norm(fmin_grad_fn(res[0]))
    return inverse_hvp, (cg_grad, status)

    


def inverse_hvp_cg(data, model, edge_index, vs, damping, device, use_torch=True):
    x_train = torch.tensor(data.train_set.nodes, device=device)
    y_train = torch.tensor(data.train_set.labels, device=device)
    inverse_hvp = []
    status = []
    cg_grad = []
    # for i, (v, p) in enumerate(zip(vs, model.parameters())):
    sizes = [p.size() for p in model.parameters() if p.requires_grad]
    # v = to_vector(vs)
    v = torch.cat([vv.view(-1) for vv in vs])
    i = None
    fmin_loss_fn = _get_fmin_loss_fn(v, model=model,
                                     x_train=x_train, y_train=y_train,
                                     edge_index=edge_index, damping=damping,
                                     sizes=sizes, p_idx=i, device=device,
                                     use_torch=use_torch)
    fmin_grad_fn = _get_fmin_grad_fn(v, model=model,
                                     x_train=x_train, y_train=y_train,
                                     edge_index=edge_index, damping=damping,
                                     sizes=sizes, p_idx=i, device=device,
                                     use_torch=use_torch)
    fmin_hvp_fn = _get_fmin_hvp_fn(v, model=model,
                                   x_train=x_train, y_train=y_train,
                                   edge_index=edge_index, damping=damping,
                                   sizes=sizes, p_idx=i, device=device,
                                   use_torch=use_torch)
    cg_callback = _get_cg_callback(v, model=model,
                                   x_train=x_train, y_train=y_train,
                                   edge_index=edge_index, damping=damping,
                                   sizes=sizes, p_idx=i, device=device,
                                   use_torch=use_torch)

    # res = minimize(fmin_loss_fn, v.view(-1), method='cg', max_iter=100)
    if use_torch:
        res = fmin_cg(
            f=fmin_loss_fn,
            x0=to_vector(vs),
            fprime=fmin_grad_fn,
            gtol=1E-4,
            # norm='fro',
            # callback=cg_callback,
            disp=False,
            full_output=True,
            maxiter=100,
        )
        # inverse_hvp.append(to_list(res[0], sizes, device)[0])
        inverse_hvp = to_list(torch.from_numpy(res[0]), sizes, device)
        cg_grad = np.linalg.norm(fmin_grad_fn(res[0]))
        status = res[4]
        # print('-----------------------------------')
        # cg_grad.append(np.linalg.norm(fmin_grad_fn(res[0]), ord=np.inf))

    else:
        res = fmin_ncg(
            f=fmin_loss_fn,
            x0=to_vector(vs),
            fprime=fmin_grad_fn,
            fhess_p=fmin_hvp_fn,
            # callback=cg_callback,
            avextol=1e-5,
            disp=False,
            full_output=True,
            maxiter=100)
        # inverse_hvp.append(to_list(res[0], sizes, device)[0])
        inverse_hvp = to_list(torch.from_numpy(res[0]), sizes, device)
        # print('-----------------------------------')
        status = res[5]
        cg_grad = np.linalg.norm(fmin_grad_fn(res[0]))

    #     x, _err, d = fmin_l_bfgs_b(
    #         func=fmin_loss_fn,
    #         x0=to_vector(v),
    #         fprime=fmin_grad_fn,
    #         iprint=0,
    #     )
    #     inverse_hvp.append(to_list(x, sizes, device)[0])
    #     status.append(d['warnflag'])
    #     err += _err.item()
    # print('error:', err, status)
    return inverse_hvp, (cg_grad, status)


def inverse_hvp_lissa(args, data, model, edge_index, v, device):
    r = args.r
    recursion_depth = args.depth
    scale = 25
    damping = 0.001

    inverse_hvps = None
    for _ in range(r):
        cur_estimate = v
        for t in range(recursion_depth):
            rand_indices = list(range(len(data.train_set)))
            random.shuffle(rand_indices)
            sampled_nodes = np.array(data.train_set.nodes)[rand_indices][:10]
            sampled_labels = np.array(data.train_set.labels)[rand_indices][:10]

            sampled_nodes = torch.tensor(sampled_nodes, device=device)
            sampled_labels = torch.tensor(sampled_labels, device=device)

            hvps, _ = hessian_vector_product(model, edge_index,
                                             sampled_nodes, sampled_labels,
                                             tuple(cur_estimate), device)

            cur_estimate = [a + (1 - damping) * b - c / scale for a, b, c in zip(v, cur_estimate, hvps)]

        if inverse_hvps is None:
            inverse_hvps = [b / scale for b in cur_estimate]
        else:
            inverse_hvps = [a + b / scale for (a, b) in zip(inverse_hvps, cur_estimate)]

    inverse_hvps = [a / r for a in inverse_hvps]
    return inverse_hvps


# def influence_sgc(args, model, data, data_prime,
#                   infected_nodes, indfected_labels,
#                   )


def influence(args, model, data, data_prime,
              infected_nodes, infected_labels,
              use_torch=True, device=torch.device('cpu'), return_norm=False):
    parameters = [p for p in model.parameters() if p.requires_grad]
    if args.transductive_edge:
        edge_index = torch.tensor(data.edges, device=device).t()
        edge_index_prime = torch.tensor(data_prime.edges, device=device).t()
    else:
        edge_index = torch.tensor(data.train_edges, device=device).t()
        edge_index_prime = torch.tensor(data_prime.train_edges, device=device).t()

    p = 1 / (data.num_train_nodes)

    # t1 = time.time()
    model.eval()
    y_hat = model(infected_nodes, edge_index_prime)
    loss1 = model.loss_sum(y_hat, infected_labels)
    g1 = grad(loss1, parameters)
    # print(f'CEU, grad new duration: {(time.time() - t1):.4f}.')

    # t1 = time.time()
    y_hat = model(infected_nodes, edge_index)
    loss2 = model.loss_sum(y_hat, infected_labels)
    g2 = grad(loss2, parameters)
    # print(f'CEU, grad old duration: {(time.time() - t1):.4f}.')

    v = [gg1 - gg2 for gg1, gg2 in zip(g1, g2)]
    # ihvp = inverse_hvp(data, model, edge_index, v, args.damping, device)
    # ihvp, (cg_grad, status) = inverse_hvp_cg_sep(data, model, edge_index, v, args.damping, device, use_torch)

    # t1 = time.time()
    if args.model == 'sgc':
        ihvp, (cg_grad, status) = inverse_hvp_cg_sgc(data, model, edge_index, v, args.lam, device)
    else:
        if len(args.hidden) == 0:
            ihvp, (cg_grad, status) = inverse_hvp_cg(data, model, edge_index, v, args.damping, device, use_torch)
        else:
            ihvp, (cg_grad, status) = inverse_hvp_cg_sep(data, model, edge_index, v, args.damping, device, use_torch)
    # print(f'CEU, hessian inverse duration: {(time.time() - t1):.4f}.')
    
    I = [- p * i for i in ihvp]

    # print('-------------------------')
    # print('Norm:', [torch.norm(ii) for ii in I])
    # print('v norm:', [torch.norm(vv) for vv in v])
    # print('CG gradient:', cg_grad)
    # print('status:', status)
    # print('-------------------------')


    if return_norm:
        return I, (torch.norm(torch.cat([i.view(-1) for i in ihvp])) ** 2) * (p ** 2)
    else:
        return I
    # return inverse_hvps, loss, status


def _update_model_weight(model, influence):
    parameters = [p for p in model.parameters() if p.requires_grad]
    with torch.no_grad():
        delta = [p + infl for p, infl in zip(parameters, influence)]
        for i, p in enumerate(parameters):
            p.copy_(delta[i])


def gcn_hessian_inv(model, edge_index, adj, k, w, nodes, y, lam, batch_size=50000, device=None):
    model.eval()
    with torch.no_grad():
        X = model.embedding(nodes)
        z = model(nodes, edge_index)[:, k]

    node_mask = torch.zeros(adj.size(0), device=device, dtype=torch.bool)
    node_mask[nodes] = 1
    adj = adj[:, node_mask][node_mask]
    X = torch.mm(adj, X)

    o = torch.sigmoid(y * z)
    D = o * (1 - o)
    H = None
    num_batch = int(math.ceil(X.size(0) / batch_size))
    for i in range(num_batch):
        lower = i * batch_size
        upper = min((i+1) * batch_size, X.size(0))
        X_i = X[lower:upper]
        if H is None:
            H = X_i.t().mm(D[lower:upper].unsqueeze(1) * X_i)
        else:
            H += X_i.t().mm(D[lower:upper].unsqueeze(1) * X_i)
    return (H + lam * X.size(0) * torch.eye(X.size(1)).float().to(device)).inverse()

# hessian of loss wrt w for binary classification
def lr_hessian_inv(model, edge_index, w, nodes, y, lam, batch_size=50000, device=None):
    '''
    The hessian here is computed wrt sum.
    input:
        w: (d,)
        X: (n,d)
        y: (n,)
        lambda: scalar
        batch_size: int
    return:
        hessian: (d,d)
    '''
    # t1 = time.time()
    model.eval()
    with torch.no_grad():
        X = model.propagate(nodes, edge_index)
    z = torch.sigmoid(y * X.mv(w))
    D = z * (1 - z)
    H = None
    num_batch = int(math.ceil(X.size(0) / batch_size))
    for i in range(num_batch):
        lower = i * batch_size
        upper = min((i + 1) * batch_size, X.size(0))
        X_i = X[lower:upper]
        if H is None:
            H = X_i.t().mm(D[lower:upper].unsqueeze(1) * X_i)
        else:
            H += X_i.t().mm(D[lower:upper].unsqueeze(1) * X_i)
    # print(f'Hessian: {(time.time() - t1):.4f}')
    # t1 = time.time()
    ih = (H + lam * X.size(0) * torch.eye(X.size(1)).float().to(device)).inverse()
    # print(f'inverse: {(time.time() - t1):.4f}')
    return ih 

def lr_grad(model, edge_index, w, nodes, y, lam):
    model.eval()
    with torch.no_grad():
        X = model.propagate(nodes, edge_index)
    z = torch.sigmoid(y * X.mv(w))
    return X.t().mv((z-1) * y) + lam * X.size(0) * w

def gcn_grad(model, edge_index, adj, k, w, nodes, y, lam, device):
    model.eval()
    with torch.no_grad():
        X = model.embedding(nodes)
        z = model(nodes, edge_index)[:, k]
    
    node_mask = torch.zeros(adj.size(0), device=device, dtype=torch.bool)
    node_mask[nodes] = 1
    # adj = adj[node_mask, node_mask]
    adj = adj[:, node_mask][node_mask]
    X = torch.mm(adj, X)

    o = torch.sigmoid(y * z)
    return X.t().mv((o-1) * y) + lam * X.size(0) * w


def get_c(delta):
    return np.sqrt(2*np.log(1.5/delta))

def get_budget(std, eps, c):
    return std * eps / c

# K = X^T * X for fast computation of spectral norm
def get_K_matrix(X):
    K = X.t().mm(X)
    return K

# using power iteration to find the maximum eigenvalue
# def sqrt_spectral_norm(A, num_iters=100):
#     '''
#     return:
#         sqrt of maximum eigenvalue/spectral norm
#     '''
#     x = torch.randn(A.size(0)).float().to(device)
#     for i in range(num_iters):
#         x = A.mv(x)
#         x_norm = x.norm()
#         x /= x_norm
#     max_lam = torch.dot(x, A.mv(x)) / torch.dot(x, x)
#     return math.sqrt(max_lam)

def sgc_edge_unlearn(args, model, data, edges_to_forget, device, mia=None):
    result = defaultdict(list)
    _model = copy.deepcopy(model).to(device)
    perm = [data.edges.index(e) for e in edges_to_forget]
    edge_index = torch.tensor(data.edges, device=device).t()
    edge_mask = torch.ones(edge_index.shape[1], dtype=torch.bool)

    train_nodes = torch.tensor(data.train_set.nodes, device=device)
    # valid_nodes = torch.tensor(data.valid_set.nodes, device=device)
    train_labels = torch.tensor(data.train_set.labels, device=device)
    # valid_labels = torch.tensor(data.valid_set.labels, device=device)

    # b = b_std * torch.randn(train_nodes.size(1), data.num_classes).float().to(device)
    b_std = 0.1

    ###########
    # budget for removal
    c_val = get_c(args.delta)
    if args.compare_gnorm:
        budget = 1e5
    else:
        if args.train_mode == 'ovr':
            budget = get_budget(b_std, args.eps, c_val) * y_train.size(1)
        else:
            budget = get_budget(b_std, args.eps, c_val)
    gamma = 1/4  # pre-computed for -logsigmoid loss
    print('Budget:', budget)

    start = time.time()
    perm_idx = 0
    for i in range(len(edges_to_forget)):
        while (edge_index[0, perm[perm_idx]] == edge_index[1, perm[perm_idx]]) or (not edge_mask[perm[perm_idx]]):
            perm_idx += 1
        edge_mask[perm[perm_idx]] = False
        source_idx = edge_index[0, perm[perm_idx]]
        dst_idx = edge_index[1, perm[perm_idx]]
        # find the other undirected edge
        rev_edge_idx = torch.logical_and(edge_index[0] == dst_idx,
                                            edge_index[1] == source_idx).nonzero().squeeze(-1)
        if rev_edge_idx.size(0) > 0:
            edge_mask[rev_edge_idx] = False
        
        perm_idx += 1

        # print(next(_model.parameters()).device, train_nodes.device, edge_index.device)
        grad_norm_approx = torch.zeros((args.num_removes, args.trails)).float()

        X_train_old = _model.propagate(train_nodes, edge_index)

        X_train = _model.propagate(train_nodes, edge_index[:, edge_mask])
        # X_valid = _model.propagate(valid_nodes, edge_index[:, edge_mask])
        y_train = F.one_hot(train_labels)
        parameters = [p for p in _model.parameters() if p.requires_grad]
        w_approx = copy.deepcopy(parameters[0]).detach()

        K = get_K_matrix(X_train).to(device)
        spec_norm = sqrt_spectral_norm(K)

        delta = torch.zeros_like(w_approx, device=device)
        for k in range(y_train.size(1)):
            y_rem = y_train[:, k]
            H_inv = cgu_utils.lr_hessian_inv(w_approx[k], X_train, y_rem, args.lam, device=device)
            # grad_i is the difference
            grad_old = cgu_utils.lr_grad(w_approx[k], X_train_old, y_rem, args.lam)
            grad_new = cgu_utils.lr_grad(w_approx[k], X_train, y_rem, args.lam)
            grad_i = grad_old - grad_new
            delta[k] = H_inv.mv(grad_i)
            # Delta = H_inv.mv(grad_i)
            Delta_p = X_train.mv(delta[k])
            # update w here. If beta exceed the budget, w_approx will be retrained
            grad_norm_approx[i] += (delta[k].norm() * Delta_p.norm() * spec_norm * gamma).cpu()
            w_approx[k] += delta[k]
            # grad_norm_approx[i, trail_iter] += (Delta.norm() * Delta_p.norm() * spec_norm * gamma).cpu()
            # if args.compare_gnorm:
            #     grad_norm_real[i, trail_iter] += lr_grad(w_approx[:, k], X_rem, y_rem, args.lam).norm().cpu()
            #     grad_norm_worst[i, trail_iter] += get_worst_Gbound_edge(args.lam, X_rem.shape[0],

        if grad_norm_approx_sum + grad_norm_approx[i] > budget:
            # retrain the model
            # grad_norm_approx_sum = 0
            # b = b_std * torch.randn(X_train.size(1), y_train.size(1)).float().to(device)
            # w_approx = ovr_lr_optimize(X_rem, y_train, args.lam, weight, b=b, num_steps=args.num_steps, verbose=args.verbose,
            #                             opt_choice=args.optimizer, lr=args.lr, wd=args.wd)
            _data
            num_retrain += 1
        else:
            grad_norm_approx_sum += grad_norm_approx[i]

        _update_model_weight(_model, [delta])

        if i+1 in args.edges:
            _data = copy.deepcopy(data)
            _data.remove_edges(edges_to_forget[:i])
            cgu_res = evaluate(args, _data, _model, device)
            if mia is not None:
                mia_result = evaluate_mia_model(mia, _model, _data, edges_to_forget[:i], device)
                _model.to(device)
                result['mia acc'].append(mia_result[0])
                result['mia auc'].append(mia_result[1])
            result['# edges'].append(i)
            result['accuracy'].append(cgu_res['accuracy'])
            result['f1'].append(cgu_res['f1'])
            result['running time'].append(time.time() - start)

    # _model.cpu()
    return result


def certified_graph_unlearn(args, model, data, data_prime, device):
    _model = copy.deepcopy(model)
    parameters = [p for p in _model.parameters() if p.requires_grad]
    
    # train_loader = DataLoader(data.train_set, shuffle=True, batch_size=args.batch)
    train_nodes = torch.tensor(data.train_set.nodes, device=device)
    train_labels = torch.tensor(data.train_set.labels, device=device)
    edge_index = torch.tensor(data.train_edges, device=device).t()
    edge_index_prime = torch.tensor(data_prime.train_edges, device=device).t()

    t0 = time.time()
    y = F.one_hot(train_labels)
    adj = torch.sparse_coo_tensor(edge_index.cpu(), torch.ones(edge_index.size(1)), 
                                  [data.num_nodes, data.num_nodes], device=device).to_dense()
    adj_prime = torch.sparse_coo_tensor(edge_index_prime, torch.ones(edge_index_prime.size(1)), 
                                        [data.num_nodes, data.num_nodes], device=device).to_dense()
    

    w = parameters[0]
    delta = torch.zeros_like(w, device=device)
    for k in range(data.num_classes):
        if args.model == 'sgc':
            # t1 = time.time()
            H_inv = lr_hessian_inv(model, edge_index_prime, w[k], train_nodes, y[:, k], args.lam, device=device)
            # print(f'hessian inverse used: {(time.time() - t1):.4f}', H_inv.size())
            # t1 = time.time()
            grad_old = lr_grad(model, edge_index, w[k], train_nodes, y[:, k], args.lam)
            # print(f'grad old: {(time.time() - t1):.4f}')
            t1 = time.time()
            grad_new = lr_grad(model, edge_index_prime, w[k], train_nodes, y[:, k], args.lam)
            # print(f'grad new: {(time.time() - t1):.4f}')
        elif args.model == 'gcn':
            H_inv = gcn_hessian_inv(model, edge_index_prime, adj_prime,
                                    k, w[k], train_nodes, y[:, k], args.lam, device=device)
            grad_old = gcn_grad(model, edge_index, adj,
                                k, w[k], train_nodes, y[:, k], args.lam, device=device)
            grad_new = gcn_grad(model, edge_index_prime, adj_prime,
                                k, w[k], train_nodes, y[:, k], args.lam, device=device)
        
        grad_i = grad_old - grad_new
        delta[k] = H_inv.mv(grad_i)
    _update_model_weight(_model, [delta])
    
    # _model.eval()
    # y_hat = _model(train_nodes, edge_index_prime)
    # loss1 = _model.loss_sum(y_hat, train_labels)
    # g1 = grad(loss1, parameters)

    # y_hat = _model(train_nodes, edge_index)
    # loss2 = _model.loss_sum(y_hat, train_labels)
    # g2 = grad(loss2, parameters)

    # v = [gg2 - gg1 for gg2, gg1 in zip(g2, g1)]
    # # ihvp = inverse_hvp(data, model, edge_index, v, args.damping, device)
    # ihvp, (cg_grad, status) = inverse_hvp_cg(data_prime, _model, edge_index_prime, v, args.damping, device, use_torch=False)
    # # dd_bound = (((torch.norm(torch.cat([i.view(-1) for i in ihvp])) ** 2) * (p ** 2))).item()
    # I = [i for i in ihvp] 
    # _update_model_weight(_model, I)
    return _model, time.time() - t0


def unlearn(args, model, data, data_prime, edges_to_forget, device, return_bound=False):
    _model = copy.deepcopy(model)
    # parameters = [p for p in _model.parameters() if p.requires_grad]
    infected_nodes = data.infected_nodes(edges_to_forget, len(args.hidden) + 1)
    infected_indices = np.where(np.in1d(np.array(data.train_set.nodes), np.array(infected_nodes)))[0]
    infected_nodes = torch.tensor(infected_nodes, device=device)
    infected_labels = torch.tensor(data.labels_of_nodes(infected_nodes.cpu().tolist()), device=device)
    # print('Number of nodes:', len(infected_nodes))

    t0 = time.time()
    if args.appr == 'cg':
        if return_bound:
            infl, bound = influence(args, _model, data, data_prime, infected_nodes, infected_labels, device=device, use_torch=True, return_norm=return_bound)
        else:
            infl = influence(args, _model, data, data_prime, infected_nodes, infected_labels, device=device, use_torch=True, return_norm=return_bound)
        # print('------------------------------------------')
        # print('infl norm:', [torch.norm(i) for i in infl])
        # print('------------------------------------------')
        # print('infl:', infl)
        # train_loader = DataLoader(data.train_set, shuffle=True, batch_size=args.batch)
        # cg_pert_influence = CGPertuabtionInfluence(
        #     model=model,
        #     objective=EdgeInfluence(data.edges, data_prime.edges, device),
        #     train_loader=train_loader,
        #     device=device,
        #     damp=0.01,
        #     maxiter=200
        # )
        # I = cg_pert_influence.influences(infected_indices)
        # infl = [- (1 / data.num_train_nodes) * i for i in I]
    else:
        edge_index = torch.tensor(data.edges, device=device).t()
        edge_index_prime = torch.tensor(data_prime.edges, device=device).t()
        infl = s_infected_nodes(model, infected_nodes, infected_labels,
                                edge_index, edge_index_prime, data.train_set, device, recusion_depth=4000)

    # print('!!!!!!', torch.linalg.norm(torch.cat([ii.view(-1) for ii in infl])))
    _update_model_weight(_model, infl)
    
    duration = time.time() - t0
    if return_bound:
        return _model, duration, bound.item(), infl
    else:
        return _model, duration


def node_influence_cg(args, _model, data, data_prime, nodes_to_forget, device):
    _nodes = torch.tensor(nodes_to_forget, device=device)
    _labels = data.labels_of_nodes(nodes_to_forget)
    _labels = torch.tensor(_labels, device=device)

    neighbors = data.neighbors(nodes_to_forget, len(args.hidden) + 1)
    neighbors_labels = data.labels_of_nodes(neighbors)
    neighbors = torch.tensor(neighbors, device=device)
    neighbors_labels = torch.tensor(neighbors_labels, device=device)

    parameters = [p for p in _model.parameters() if p.requires_grad]
    edge_index = torch.tensor(data.train_edges, device=device).t()
    edge_index_prime = torch.tensor(data_prime.train_edges, device=device).t()
    p = 1 / data.num_train_nodes

    _model.eval()
    # target nodes:
    y_hat = _model(_nodes, edge_index)
    loss = _model.loss_sum(y_hat, _labels)
    grad_nodes = grad(loss, parameters)

    if len(neighbors) != 0:
        y_hat = _model(neighbors, edge_index_prime)
        loss = _model.loss_sum(y_hat, neighbors_labels)
        g1 = grad(loss, parameters)

        y_hat = _model(neighbors, edge_index)
        loss = _model.loss_sum(y_hat, neighbors_labels)
        g2 = grad(loss, parameters)

        v = [gg1 - gg2 - gg_n for gg1, gg2, gg_n in zip(g1, g2, grad_nodes)]
        ihvp, cg_grad, status = inverse_hvp_cg(data, _model, edge_index, v, args.damping, device, use_torch=False)
    else:
        ihvp, _, _ = inverse_hvp_cg(data, _model, edge_index, grad_nodes, args.damping, device, use_torch=False)

    I = [- p * i for i in ihvp]
    return I


def unlearn_node(args, model, data, data_prime, nodes_to_forget, device):
    _model = copy.deepcopy(model)
    t0 = time.time()
    infl = node_influence_cg(args, _model, data, data_prime, nodes_to_forget, device)
    _update_model_weight(_model, infl)
    duration = time.time() - t0
    return _model, duration

# if __name__ == '__main__':
#     embedding_size = 3
#     num_nodes = 100

#     v = [
#         torch.rand(num_nodes, embedding_size, device=torch.device('cuda:0')),
#         torch.rand(embedding_size, 3, device=torch.device('cuda:0')),
#         torch.rand(3, device=torch.device('cuda:0')),
#     ]
#     sizes = [vv.size() for vv in v]

#     print('v:', v)
#     vec = to_vector(v)
#     print('vector:', vec)
#     l = to_list(vec, sizes, torch.device('cpu'))
#     print('list', l)
# if __name__ == '__main__':
    # parser = argument_parser()
    # args = parser.parse_args()

    # data = load_data(args)
    # device = torch.device(f'cuda:{args.gpu}') if torch.cuda.is_available() and args.gpu >= 0 else torch.device('cpu')

    # original_model = train_model(args, data, eval=False, verbose=False, device=device)

    # edges_to_forget = sample_edges(args, data)[:100]
    # _data = copy.deepcopy(data)
    # _data.remove_edges(edges_to_forget)
    # compare_influence_functions(args, original_model, data, _data, edges_to_forget, device)
