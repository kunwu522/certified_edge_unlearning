from collections import defaultdict
from functools import reduce
from pickletools import optimize
import numpy as np
import time
import math
import torch
import copy
import random
from torch.utils.data import DataLoader
from torch.autograd import grad
from tqdm import tqdm
from train import test
from scipy.optimize import fmin_ncg, fmin_l_bfgs_b, fmin_cg
from torchmin import minimize
from utils import create_model, edges_remove_nodes, load_model, remove_undirected_edges, sample_nodes, loss_of_test_nodes, save_model, train_set_remove_nodes
from hessian import hessian_vector_product, hessian


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
        total = reduce(lambda a, b: a*b, size)
        result.append(torch.from_numpy(_v[:total].reshape(size)).float().to(device))
        _v = _v[total:]
    return tuple(result)
    # return torch.tensor(v.reshape(sizes[0]), dtype=torch.float, device=device)


def _mini_batch_hvp(x, **kwargs):
    model = kwargs['model']
    train_loader = kwargs['train_loader']
    edge_index = kwargs['edge_index']
    # edge_index_prime = kwargs['edge_index_prime'],
    damping = kwargs['damping']
    device = kwargs['device']
    sizes = kwargs['sizes']
    p_idx = kwargs['p_idx']

    hvp = None
    # x = to_list(x, sizes, device)
    for nodes, labels in train_loader:
        nodes = nodes.to(device)
        labels = labels.to(device)
        _hvp, L = hessian_vector_product(model, edge_index, nodes, labels, (x.view(sizes[0]),), device, p_idx)
        if hvp is None:
            hvp = [b for b in _hvp]
        else:
            hvp = [a + b for a, b in zip(hvp, _hvp)]
    # return [a + b * damping for (a, b) in zip(hvp, x)]
    # return hvp[0].view(-1)
    return hvp[0].view(-1) + damping * x, L


def _hessain_hvp(x, **kwargs):
    # print('x:', x.shape)
    damping = kwargs['damping']
    device = kwargs['device']
    B = kwargs['H'].view(x.shape[0], x.shape[0])
    # B = torch.cat([b.view(-1) for b in B]).view(x.shape[0], x.shape[0]).to(device)

    # xx = torch.from_numpy(x).to(device)
    # hvp = np.dot(B.reshape(int(math.sqrt(total)), int(math.sqrt(total))), x)
    hvp = torch.mm(B, x.view(-1, 1)).view(-1)
    return hvp + damping * x


def _get_fmin_loss_fn2(v, **kwargs):
    device = kwargs['device']

    def get_fmin_loss(x):
        with torch.no_grad():
            x = torch.tensor(x, dtype=torch.float, device=device)
            hvp, _ = _mini_batch_hvp(x, **kwargs)
            obj = torch.norm(hvp - v)
        return obj.detach().cpu().numpy()

    return get_fmin_loss


def _get_fmin_prime_fn(v, **kwargs):
    device = kwargs['device']

    def get_fmin_grad(x):
        x = torch.tensor(x, dtype=torch.float, device=device)
        hvp, _ = _mini_batch_hvp(x, **kwargs)

        return to_vector(hvp - v.view(-1))


def _get_fmin_loss_fn(v, **kwargs):
    device = kwargs['device']

    def get_fmin_loss(x):
        x = torch.tensor(x, dtype=torch.float, device=device)
        if kwargs['H'] is None:
            hvp, L = _mini_batch_hvp(x, **kwargs)
            obj = 0.5 * torch.dot(hvp, x) - torch.dot(v.view(-1), x)
            return obj.detach().cpu().numpy()
        else:
            # _v = torch.from_numpy(v).to(device)
            hvp = _hessain_hvp(x, **kwargs)
            obj = 0.5 * torch.dot(hvp, x) - torch.dot(v.view(-1), x)
            return to_vector(obj)
    return get_fmin_loss


def _get_fmin_grad_fn(v, **kwargs):
    device = kwargs['device']

    def get_fmin_grad(x):
        x = torch.tensor(x, dtype=torch.float, device=device)
        if kwargs['H'] is None:
            hvp, _ = _mini_batch_hvp(x, **kwargs)
            return to_vector(hvp - v.view(-1))
        else:
            hvp = _hessain_hvp(x, **kwargs)
            return to_vector(hvp - v.view(-1))
    return get_fmin_grad


def _get_fmin_hvp_fn(v, **kwargs):
    device = kwargs['device']

    def get_fmin_hvp(x, p):
        p = torch.tensor(p, dtype=torch.float, device=device)
        if kwargs['H'] is None:
            hvp, _ = _mini_batch_hvp(p, **kwargs)
            return to_vector(hvp)
        else:
            hvp = _hessain_hvp(p, **kwargs)
            return to_vector(hvp)
    return get_fmin_hvp


def _get_cg_callback(v, **kwargs):
    device = kwargs['device']

    def cg_callback(x):
        x = torch.tensor(x, dtype=torch.float, device=device)
        hvp, L = _mini_batch_hvp(x, **kwargs)
        obj = 0.5 * torch.dot(hvp, x) - torch.dot(v.view(-1), x)
        g = to_vector(hvp - v.view(-1))
        print(f'loss: {obj:.4f}, grad: {(np.linalg.norm(g) / g.size):.8f}')
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
            maxiter=100)

        inverse_hvp.append(to_list(res[0], sizes, device)[0])
        status.append(res[5])
    return inverse_hvp, tuple(status)


def _influence_gd(args, Hs, bs, device):
    num_iters = 200

    result = []
    for H, b in zip(Hs, bs):
        print('!!!!!!!!', H.size(), b.size())
        _size = b.size()
        b = b.squeeze().view(-1, 1)
        H = H.squeeze().view(b.size(0), b.size(0))
        # if len(b) <= 10000:
        #     H = H.to(device)
        # else:
        #     b = b.cpu()
        H = H.to(device)
        # print(torch.matrix_rank(H), H.shape)

        x = torch.zeros_like(b)
        if args.model == 'gin':
            lr = 1E-9
        else:
            lr = 1E-6
        # optimizer = torch.optim.SGD([x], lr=0.001)

        for i in range(num_iters):
            # optimizer.zero_grad()
            # f.backward()
            # optimizer.step()
            g = torch.mm(torch.mm(H.t(), H), x) - torch.mm(H.t(), b)
            x = x - lr * g

            f = torch.norm(torch.mm(H, x.view(-1, 1)) - b) ** 2
            print(f'Iter {i}, loss: {f:.4f}.')
            # print('x', x)
        result.append(x.detach().view(_size))
        torch.cuda.empty_cache()
    return result


def inverse_hvp_cg(data, model, edge_index, vs, damping, device, H=None):
    train_loader = DataLoader(data['train_set'], batch_size=int(len(data['train_set'])/1), shuffle=True)

    inverse_hvp = []
    status = []
    cg_loss = []
    err = 0.
    # for i, hv in enumerate(zip(H, vs)):
    # h, v = hv
    # h = h.to(device)
    for i, (v, p) in enumerate(zip(vs, model.parameters())):
        h = None
        sizes = [v.size()]
        fmin_loss_fn = _get_fmin_loss_fn(v, model=model, edge_index=edge_index,
                                         train_loader=train_loader, damping=damping,
                                         sizes=sizes, p_idx=i, device=device,
                                         H=h, p=p.data.detach())
        fmin_grad_fn = _get_fmin_grad_fn(v, model=model, edge_index=edge_index,
                                         train_loader=train_loader, damping=damping,
                                         sizes=sizes, p_idx=i, device=device,
                                         H=h, p=p.data.detach())
        fmin_hvp_fn = _get_fmin_hvp_fn(v, model=model, edge_index=edge_index,
                                       train_loader=train_loader, damping=damping,
                                       sizes=sizes, p_idx=i, device=device,
                                       H=h, p=p.data.detach())
        cg_callback = _get_cg_callback(v, model=model, edge_index=edge_index,
                                       train_loader=train_loader, damping=damping,
                                       sizes=sizes, p_idx=i, device=device,
                                       H=h, p=p.data.detach())

        # res = minimize(fmin_loss_fn, v.view(-1), method='cg', max_iter=100)

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
        inverse_hvp.append(to_list(res[0], sizes, device)[0])
        # print('-----------------------------------')
        status.append(res[4])
        cg_loss.append(res[1])

        # res = fmin_ncg(
        #     f=fmin_loss_fn,
        #     x0=to_vector(v),
        #     fprime=fmin_grad_fn,
        #     fhess_p=fmin_hvp_fn,
        #     callback=cg_callback,
        #     avextol=1e-5,
        #     disp=False,
        #     full_output=True,
        #     maxiter=100)
        # inverse_hvp.append(to_list(res[0], sizes, device)[0])
        # print('-----------------------------------')
        # status.append(res[5])
        # cg_loss.append(res[1])

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
    return inverse_hvp, np.mean(cg_loss), status


def inverse_hvp_lissa(args, data, model, edge_index, v, device):
    r = args.r
    recursion_depth = args.depth
    scale = 25
    damping = 0.001

    inverse_hvps = None
    for _ in range(r):
        cur_estimate = v
        for t in range(recursion_depth):
            rand_indices = list(range(len(data['train_set'])))
            random.shuffle(rand_indices)
            sampled_nodes = np.array(data['train_set'].nodes)[rand_indices][:10]
            sampled_labels = np.array(data['train_set'].labels)[rand_indices][:10]

            sampled_nodes = torch.tensor(sampled_nodes, device=device)
            sampled_labels = torch.tensor(sampled_labels, device=device)

            hvps, _ = hessian_vector_product(model, edge_index,
                                             sampled_nodes, sampled_labels,
                                             tuple(cur_estimate), device)

            cur_estimate = [a + (1-damping) * b - c/scale for a, b, c in zip(v, cur_estimate, hvps)]

        if inverse_hvps is None:
            inverse_hvps = [b/scale for b in cur_estimate]
        else:
            inverse_hvps = [a + b / scale for (a, b) in zip(inverse_hvps, cur_estimate)]

    inverse_hvps = [a / r for a in inverse_hvps]
    return inverse_hvps


def influence_node(args, data, train_node, test_node=None, device=torch.device('cpu')):
    model = load_model(args, data).to(device)

    r = args.r
    recursion_depth = args.depth
    scale = args.scale

    edge_index = torch.tensor(data['edges'], device=device).t()

    parameters = [p for p in model.parameters() if p.requires_grad]

    model.eval()

    x = torch.tensor([train_node[0]], device=device)
    y = torch.tensor([train_node[1]], device=device)
    y_hat = model(x, edge_index)
    loss = model.loss_sum(y_hat, y)
    grad_train = grad(loss, parameters)

    if test_node is not None:
        y_hat = model(torch.tensor([test_node['node']], device=device), edge_index)
        loss_test = model.loss(y_hat, torch.tensor([test_node['label']], device=device))
        grad_loss_test = grad(loss_test, parameters)
    v = grad_train if test_node is None else grad_loss_test

    inverse_hvps = None
    for _ in range(r):
        cur_estimate = v
        for t in range(recursion_depth):
            sampled_node, sampled_label = sample_nodes(args, data, method='random', num_nodes=-1)
            sampled_node = torch.tensor([sampled_node], device=device)
            sampled_label = torch.tensor([sampled_label], device=device)
            # sampled_node = torch.tensor(data['train_set'].nodes, device=device)
            # sampled_label = torch.tensor(data['train_set'].labels, device=device)

            hvps = _hessian_vector_product(model, edge_index,
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
        return np.sum([torch.sum((inverse_hvp) * g).cpu().item() for inverse_hvp, g in zip(inverse_hvps, grad_train)]) / data['num_nodes']


def _influence(args, data, model, parameters, edges_prime, device=torch.device('cpu')):
    edge_index = torch.tensor(data['edges'], device=device).t()
    edge_index_prime = torch.tensor(edges_prime, dtype=torch.long, device=device).t()

    model.eval()
    x_train = torch.tensor(data['train_set'].nodes, device=device)
    y_train = torch.tensor(data['train_set'].labels, device=device)
    y_hat = model(x_train, edge_index_prime)
    loss_prime = model.loss_sum(y_hat, y_train)
    v = grad(loss_prime, parameters)
    H = None

    # t0 = time.time()
    # H, _ = hessian(model, edge_index_prime, x_train, y_train, parameters)
    # print(f'Finished Hessian in {(time.time() - t0):.2f}s.')

    # influence = _influence_gd(args, H, v, device)
    # return influence

    # inverse_hvps = inverse_hvp_lissa(args, data, model, edge_index_prime, v, device)
    # return inverse_hvps

    # Directly approximate of H-1v
    # t0 = time.time()
    influence, loss, status = inverse_hvp_cg(data, model, edge_index_prime, v, args.damping, device, H)
    # print(f'Finished CG in {(time.time() - t0):.2f}s.')
    return influence

    # return inverse_hvps, loss, status


def _influence_new(args, data, edges_to_forget, test_node=None, device=torch.device('cpu')):
    model = load_model(args, data).to(device)
    parameters = [p for p in model.parameters() if p.requires_grad]

    edges_ = [(v1, v2) for v1, v2 in data['edges'] if (v1, v2) not in edges_to_forget]
    # edges_.remove(edge_to_forget)
    # edges_ =
    edge_index = torch.tensor(data['edges'], device=device).t()
    edge_index_prime = torch.tensor(edges_, dtype=torch.long, device=device).t()

    model.eval()

    x_train = torch.tensor(data['train_set'].nodes, device=device)
    y_train = torch.tensor(data['train_set'].labels, device=device)
    y_hat = model(x_train, edge_index_prime)
    loss_prime = model.loss_sum(y_hat, y_train)
    v = grad(loss_prime, parameters)

    # Directly approximate of H-1v
    inverse_hvps, loss, status = inverse_hvp_cg(data, model, edge_index_prime, v, args.damping, device)

    # Calculate H first and approximate H-1v
    # hessian_path = os.path.join(
    #     './data/', args.data, f'hessian_{args.model}_e{edge_to_forget[0]}-{edge_to_forget[1]}.list')
    # if os.path.exists(hessian_path):
    #     with open(hessian_path, 'rb') as fp:
    #         H = pickle.load(fp)
    # else:
    #     t0 = time.time()
    #     H = hessian(model, edge_index_prime, x_train, y_train, device)
    #     print(f'Calculate hessian on G without {edge_to_forget}, duration {(time.time()-t0):.4f}.')
    #     # with open(hessian_path, 'wb') as fp:
    #     #     pickle.dump([h.cpu().tolist() for h in H], fp)
    # inverse_hvps, status = inverse_hvp_cg_hessian(H, v, args.damping, device)

    if test_node is not None:
        y_hat = model(torch.tensor([test_node['node']], device=device), edge_index)
        loss_test = model.loss(y_hat, torch.tensor([test_node['label']], device=device))
        grad_loss_test = grad(loss_test, parameters)

        return np.sum([torch.sum(inverse_hvp * g).cpu().item() for inverse_hvp, g in zip(inverse_hvps, grad_loss_test)])
    else:
        return inverse_hvps, loss, status


def influence(args, data, edge_to_forget, test_node=None, device=torch.device('cpu')):
    model = load_model(args, data).to(device)

    # with open(os.path.join('./data', args.data, 'edge_epsilon.dict'), 'rb') as fp:
    #     edge2epsilon = pickle.load(fp)

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
        y_hat = model(torch.tensor([test_node['node']], device=device), edge_index_prime)
        loss_test = model.loss(y_hat, torch.tensor([test_node['label']], device=device))
        grad_loss_test = grad(loss_test, parameters)
    v = grad_delta_e if test_node is None else grad_loss_test
    # v = grad_delta_e

    if args.approx == 'lissa':
        inverse_hvps = inverse_hvp_lissa(args, data, model, edge_index, v, device)
    elif args.approx == 'cg':
        inverse_hvps, status = inverse_hvp_cg(data, model, edge_index, v, args.damping, device)

    if test_node is None:
        return inverse_hvps, status
    else:
        return np.sum([torch.sum((inverse_hvp) * g).cpu().item() for inverse_hvp, g in zip(inverse_hvps, grad_delta_e)])
        # return np.sum([torch.sum(g * inverse_hvp).cpu().item() for inverse_hvp, g in zip(inverse_hvps, grad_loss_test)]) * edge2epsilon[edge_to_forget] / len(data['train_set'])


def _update_model_weight(parameters, inverse_hvps):
    with torch.no_grad():
        # delta = [p + infl for p, infl in zip(parameters, inverse_hvps)]
        delta = [p - infl for p, infl in zip(parameters, inverse_hvps)]
        for i, p in enumerate(parameters):
            p.copy_(delta[i])


def batch_unlearn(args, data, edges_to_forget, num_edges_list, device):
    t0 = time.time()

    acc, f1, losses = [], [], []
    unlearn_time = []
    for num_edge in tqdm(num_edges_list, desc='unlearn'):
        model = load_model(args, data).to(device)
        parameters = [p for p in model.parameters() if p.requires_grad]
        edges_ = copy.deepcopy(data['edges'])
        for e in edges_to_forget[:num_edge]:
            edges_.remove(e)
        # edges_ = [(v1, v2) for v1, v2 in data['edges'] if (v1, v2) not in edges_to_forget[:num_edge]]
        # edge_index = torch.tensor(data['edges'], device=device).t()

        t0 = time.time()
        # BU
        if args.unlearn_batch_size is not None:
            num_batchs = math.ceil(len(edges_to_forget) / args.unlearn_batch_size)
            cg_losses = []
            status_count = defaultdict(int)
            for b in range(num_batchs):
                batch_edges = edges_to_forget[b * args.unlearn_batch_size: (b + 1) * args.unlearn_batch_size]
                inverse_hvps, loss, status = _influence(args, data, model, parameters, batch_edges, device=device)
                cg_losses.append(loss)
                for s in status:
                    status_count[s] += 1
            _update_model_weight(parameters, inverse_hvps)
            if args.verbose:
                print(f'Batch unlearning done. CG loss: {np.sum(cg_losses):.4f}, status: {dict(status_count)}.')

        else:  # DU
            # inverse_hvps, cg_loss, status = _influence_new(args, data, edges_, device=device)
            inverse_hvps, cg_loss, status = _influence(
                args, data, model, parameters, edges_, device=device)
            status_count = defaultdict(int)
            for s in status:
                status_count[s] += 1
            _update_model_weight(parameters, inverse_hvps)
            if args.verbose:
                print(f'Direct unlearning done. CG loss: {cg_loss:.4f}, status: {dict(status_count)}.')
        duration = time.time() - t0
        # edge_index_prime = torch.tensor(edges_, dtype=torch.long, device=device).t()

        # model.eval()

        # x_train = torch.tensor(data['train_set'].nodes, device=device)
        # y_train = torch.tensor(data['train_set'].labels, device=device)
        # y_hat = model(x_train, edge_index_prime)
        # loss_prime = model.loss_sum(y_hat, y_train)
        # v = grad(loss_prime, parameters)

        # # Directly approximate of H-1v
        # inverse_hvps, loss, status = inverse_hvp_cg(data, model, edge_index_prime, v, args.damping, device)
        # with torch.no_grad():
        #     delta = [p + infl for p, infl in zip(parameters, inverse_hvps)]
        #     # delta = [p + (infl/num_edge) for p, infl in zip(parameters, inverse_hvps)]
        #     # delta = [p + infl / data['num_nodes'] for p, infl in zip(parameters, inverse_hvps)]
        #     for i, p in enumerate(parameters):
        #         p.copy_(delta[i])
        if args.save:
            save_model(args, model, type='unlearn', edges=num_edge)
        else:
            test_loader = DataLoader(data['test_set'], batch_size=args.test_batch, shuffle=False)
            edge_index_prime = torch.tensor(edges_, device=device).t()
            res, loss = test(model, test_loader, edge_index_prime, device)
            acc.append(res['accuracy'])
            f1.append(res['macro avg']['f1-score'])
            losses.append(loss)
            unlearn_time.append(duration)

    return acc, f1, losses, unlearn_time
    print(f'Unlearning finsihed, duration: {(time.time() - t0):.4f}s.')


def unlearn(args, data, model, edges_to_forget, device, influence=False):
    _model = copy.deepcopy(model)
    parameters = [p for p in _model.parameters() if p.requires_grad]
    _edges = remove_undirected_edges(data['edges'], edges_to_forget)

    t0 = time.time()
    # inverse_hvps, cg_loss, status = _influence(
    #     args, data, _model, parameters, _edges, device=device)
    # inverse_hvps = _influence(args, data, _model, parameters, _edges, device=device)
    # status_count = defaultdict(int)
    # for s in status:
    #     status_count[s] += 1
    infl = _influence(args, data, _model, parameters, _edges, device=device)
    _update_model_weight(parameters, infl)
    # if args.verbose:
    #     print(f'direct unlearning done. cg loss: {cg_loss:.4f}, status: {dict(status_count)}.')
    duration = time.time() - t0
    if influence:
        return _model, duration, [torch.linalg.norm(i.detach().cpu()) for i in infl]
    else:
        return _model, duration


def node_unlearn(args, data, model, nodes_to_forget, device):
    t0 = time.time()

    _model = copy.deepcopy(model)
    parameters = [p for p in _model.parameters() if p.requires_grad]

    _data = train_set_remove_nodes(data, nodes_to_forget)
    _edges = edges_remove_nodes(data['edges'], nodes_to_forget)
    inverse_hvps, cg_loss, status = _influence(
        args, _data, _model, parameters, _edges, device=device)
    status_count = defaultdict(int)
    for s in status:
        status_count[s] += 1
    _update_model_weight(parameters, inverse_hvps)
    if args.verbose:
        print(f'direct unlearning done. cg loss: {cg_loss:.4f}, status: {dict(status_count)}.')
    duration = time.time() - t0
    return _model, duration


def _unlearn(args, data, edges_to_forget, num_edges_list, device, verbose=False):
    ''' Deprecated of unlearning.
        Unlearning edges one by one has proved that has very poor performance.
        Therefore, it is deprecated and replaced by a new unlearn function.
        (changed the name to _unlearn())
    '''
    t0 = time.time()

    model = load_model(args, data).to(device)
    parameters = [p for p in model.parameters() if p.requires_grad]

    min_num_edges = min(num_edges_list)
    cg_loss = 0.
    status_count = defaultdict(int)
    pbar = tqdm(edges_to_forget, desc=f'unlearning {min_num_edges} edges', total=min_num_edges)
    for i, edge in enumerate(pbar):
        # if i in [930, 994, 1202, 1290]:
        #     continue
        if i in num_edges_list:
            print(f'The loss of CG: {(cg_loss / i):.4f}')
            print(f'Unlearned {i} edges, status_count:', status_count)
            save_model(args, model, type='unlearn', edges=i)
            num_edges_list.remove(i)
            if num_edges_list:
                min_num_edges = min(num_edges_list)
                pbar.total = min_num_edges
                pbar.set_description(f'unlearning {min_num_edges} edges')

        if len(num_edges_list) == 0:
            break

        # inverse_hvps, status = influence(args, data, edge, device=device)
        inverse_hvps, loss, status = _influence_new(args, data, edge, device=device)
        # inverse_hvps, loss, status = _influence(args, data, model, parameters, edge, device=device)
        cg_loss += loss.item()
        # print()
        # print('---------------------------------------------------------------')
        # print(inverse_hvps)
        # print('---------------------------------------------------------------')
        for s in status:
            status_count[s] += 1
        with torch.no_grad():
            delta = [p + infl for p, infl in zip(parameters, inverse_hvps)]
            # delta = [p + infl / data['num_nodes'] for p, infl in zip(parameters, inverse_hvps)]
            for i, p in enumerate(parameters):
                p.copy_(delta[i])

    print(f'Unlearning finsihed, duration: {(time.time() - t0):.4f}s.')


def influences_node(args, data, nodes_to_forget, device):
    print('Start to unlearn...')
    test_node = loss_of_test_nodes(args, data, device=device)[0]
    print('test_node:', test_node)

    results = {}
    for node in tqdm(nodes_to_forget, desc='  influences'):
        infl = influence_node(args, data, node, test_node=test_node, device=device)
        print(f'The influence of edge {node} is {infl:.4f}.')
        results[node] = infl

    print('Unlearning finished.')
    print()
    return results


def influences(args, data, edges_to_forget, device):
    print('Start to unlearn...')
    test_node = loss_of_test_nodes(args, data, device=device)[0]
    print('test_node:', test_node)

    edges_ = [(v1, v2) for v1, v2 in data['edges']]

    results = {}
    for edge in tqdm(edges_to_forget, desc='  influences'):
        edges_.remove(edge)
        # infl = influence(args, data, edge, test_node=test_node, device=device)
        infl = _influence_new(args, data, edge, test_node=test_node, device=device)
        # print(f'The influence of edge {edge} is {infl:.4f}.')
        results[edge] = infl
        edges_ = [(v1, v2) for v1, v2 in data['edges']]

    print('Unlearning finished.')
    print()
    return results


# if __name__ == '__main__':
#     embedding_size = 3
#     num_nodes = 10

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


if __name__ == '__main__':
    device = torch.device('cpu')

    N = 50
    d = 32
    k = 3

    H_e = torch.rand(N * d, N * d)
    B_e = torch.mm(H_e, H_e.T)
    v_e = torch.rand(N, d)

    H_w = torch.rand(d * k, d * k)
    B_w = torch.mm(H_w, H_w.T)
    v_w = torch.rand(d, k)

    H_b = torch.rand(k, k)
    B_b = torch.mm(H_b, H_b.T)
    v_b = torch.rand(k)

    def get_fmin_loss_fn(Bs, vs):
        sizes = [v.size() for v in vs]

        def get_fmin_loss(x):
            hvp = [torch.mm(xx.view(1, -1), B) for B, xx in zip(Bs, to_list(x, sizes, device))]
            return 0.5 * np.dot(to_vector(hvp), x) - np.dot(to_vector(vs), x)
        return get_fmin_loss

    def get_fmin_grad_fn(Bs, vs):
        sizes = [v.size() for v in vs]

        def get_fmin_grad(x):
            hvp = [torch.mm(B, xx.view(-1, 1)) for B, xx in zip(Bs, to_list(x, sizes, device))]
            return to_vector(hvp) - to_vector(vs)
        return get_fmin_grad

    def get_fmin_hvp_fn(Bs, vs):
        sizes = [v.size() for v in vs]

        def get_fmin_hvp(x, p):
            hvp = [torch.mm(B, pp.view(-1, 1)) for B, pp in zip(Bs, to_list(p, sizes, device))]
            return to_vector(hvp)
        return get_fmin_hvp

    fmin_loss_fn = get_fmin_loss_fn((B_e, B_w, B_b), (v_e, v_w, v_b))
    fmin_grad_fn = get_fmin_grad_fn((B_e, B_w, B_b), (v_e, v_w, v_b))
    fmin_hvp_fn = get_fmin_hvp_fn((B_e, B_w, B_b), (v_e, v_w, v_b))
    res = fmin_ncg(
        f=fmin_loss_fn,
        x0=to_vector([v_e, v_w, v_b]),
        fprime=fmin_grad_fn,
        fhess_p=fmin_hvp_fn,
        avextol=1e-8,
        disp=True,
        full_output=True,
        maxiter=100
    )
    inverse_hvp = res[0]
    status = res[5]
    print('Status:', status)

    print('Inverse_hvp:', [inverse_hvp[: N * d].reshape(N, d),
          inverse_hvp[N*d: N*d + d*k].reshape(d, k), inverse_hvp[-k:]])

    # H = F.hessian(dummy_gcn, (v_e, v_w, v_b))
    # H_e = H[0][0].view(10, 10).numpy()
    # H_w = H[1][1].view(15, 15).numpy()
    # H_b = H[2][2].numpy()
    ihv_e = np.dot(np.linalg.inv(B_e), v_e.view(-1).numpy())
    ihv_w = np.dot(np.linalg.inv(B_w), v_w.view(-1).numpy())
    ihv_b = np.dot(np.linalg.inv(B_b), v_b.view(-1).numpy())

    print('Truth:', [ihv_e, ihv_w, ihv_b])
