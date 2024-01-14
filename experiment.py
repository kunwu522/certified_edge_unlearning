'''
Experiment of CEU.

Kun Wu
Stevens Institute of Technology
'''

from collections import defaultdict
from email.policy import default
import os
import copy
import time
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from argument import argument_parser
from data_loader import load_data
from mia import train_mia, evaluate_mia_model
from train import evaluate, train_model
from retrain import retrain
from unlearn import unlearn, sgc_edge_unlearn
from utils import sample_edges


def _norm_graident(model, data):
    nodes = torch.tensor(data.train_set.nodes, device=device)
    labels = torch.tensor(data.train_set.labels, device=device)
    edge_index = torch.tensor(data.edges, device=device).t()

    model.eval()
    y_hat = model(nodes, edge_index)
    loss = model.loss(y_hat, labels)
    g = torch.autograd.grad(loss, [p for p in model.parameters() if p.requires_grad])
    gr = torch.linalg.norm(torch.cat(([gg.view(-1) for gg in g]))).item()
    return gr

def _certified_upper_bound(args, data, edges_to_forget):
    infected_nodes = data.infected_nodes(edges_to_forget, len(args.hidden) + 1)
    edge_index = np.array(edges_to_forget).T
    n_v = []
    for v in infected_nodes:
        n_v.append(np.sum(edge_index[0] == v) + np.sum(edge_index[1] == v))

    return args.gamma1 * (args.gamma2 ** 2) / ((args.lam ** 4) * data.num_train_nodes) * (sum(n_v) ** 2)

def _gradient_residual(args, data, device):
    args.edges = [100, 200, 400, 600, 800, 1000] if args.data != 'cs' else [1000, 2000, 4000, 6000, 8000, 10000]

    result = defaultdict(list)
    for _ in tqdm(range(10), desc=f'{args.data}-{args.model}'):
        original_model = train_model(args, data, eval=False, verbose=False, device=device)
        original_res = evaluate(args, data, original_model, device)
        gr = _norm_graident(original_model, data)
        result['# edges'].append(0)
        result['retrain'].append(original_res['accuracy'])
        result['unlearn'].append(original_res['accuracy'])
        result['gradient residual'].append(gr)
        result['data dependent bound'].append(0)
        result['certified'].append(0)

        edges_to_forget = sample_edges(args, data, method=args.method)
        for num_edges in args.edges:
            _data = copy.deepcopy(data)
            _data.remove_edges(edges_to_forget[:num_edges])
            
            retrain_model, _ = retrain(args, _data, device)
            retrain_res = evaluate(args, _data, retrain_model, device)
            unlearn_model, _, db = unlearn(args, original_model, data, _data, edges_to_forget[:num_edges], device, return_bound=True)
            unlearn_res = evaluate(args, _data, unlearn_model, device)
            gr = _norm_graident(unlearn_model, _data)

            result['# edges'].append(num_edges)
            result['retrain'].append(retrain_res['accuracy'])
            result['unlearn'].append(unlearn_res['accuracy'])
            result['gradient residual'].append(gr)
            result['data dependent bound'].append(db)

            ub = _certified_upper_bound(args, data, edges_to_forget[:num_edges])
            result['certified'].append(ub)

    df = pd.DataFrame(result)
    print(df.groupby('# edges').mean())
    result_path = f'./result/bound_{args.data}_{args.model}_{args.method}'
    if args.feature:
        result_path += '_feature'
    if len(args.hidden) > 0:
        result_path += f'_l{len(args.hidden)}'
    if args.add_noise:
        result_path += '_noise'
    timestamp = time.time()
    result_path += f'_{timestamp}.csv'
    df.to_csv(result_path)


def _unlearning(args, data, device):
    result = defaultdict(list)
    args.edges = [100, 200, 400, 600, 800, 1000] if args.data != 'cs' else [1000, 2000, 4000, 6000, 8000, 10000]

    for _ in tqdm(range(args.num_trials), desc=f'{args.data}-{args.model}'):
        o_acc = 0
        while o_acc < 0.7:
            data = load_data(args)
            original_model = train_model(args, data, eval=False, verbose=False, device=device)
            original_res = evaluate(args, data, original_model, device)
            o_acc = original_res['accuracy']

        if len(args.hidden) == 0:
            original_grn = _norm_graident(original_model, data)

        # add original result as 0 edges
        result['# edges'].append(0)
        result['setting'].append('retrain')
        result['accuracy'].append(original_res['accuracy'])
        result['f1'].append(original_res['f1'])
        result['running time'].append(0)
        if len(args.hidden) == 0:
            result['grn'].append(original_grn)

        result['# edges'].append(0)
        result['setting'].append('unlearn')
        result['accuracy'].append(original_res['accuracy'])
        result['f1'].append(original_res['f1'])
        result['running time'].append(0)
        if len(args.hidden) == 0:
            result['grn'].append(original_grn)

        # Retraining and unlearning
        edges_to_forget = sample_edges(args, data, method=args.method)
        for num_edges in args.edges:
            _data = copy.deepcopy(data)
            _data.remove_edges(edges_to_forget[:num_edges])

            retrain_model, retrain_time = retrain(args, _data, device, verbose=args.verbose)
            unlearn_model, unlearn_time = unlearn(args, original_model, data, _data, edges_to_forget[:num_edges], device)
            retrain_res = evaluate(args, _data, retrain_model, device)
            unlearn_res = evaluate(args, _data, unlearn_model, device)
            if len(args.hidden) == 0:
                retrain_grn = _norm_graident(retrain_model, _data)
                unlearn_grn = _norm_graident(unlearn_model, _data)

            result['# edges'].append(num_edges)
            result['setting'].append('retrain')
            result['accuracy'].append(retrain_res['accuracy'])
            result['f1'].append(retrain_res['f1'])
            result['running time'].append(retrain_time)
            if len(args.hidden) == 0:
                result['grn'].append(retrain_grn)

            result['# edges'].append(num_edges)
            result['setting'].append('unlearn')
            result['accuracy'].append(unlearn_res['accuracy'])
            result['f1'].append(unlearn_res['f1'])
            result['running time'].append(unlearn_time)
            if len(args.hidden) == 0:
                result['grn'].append(unlearn_grn)

    df = pd.DataFrame(data=result)
    result_path = os.path.join('./result', f'unlearn_{args.data}_{args.model}_{args.method}')
    if args.feature:
        result_path += '_feature'
    if len(args.hidden) > 0:
        result_path += f'_l{len(args.hidden)}'
    if args.add_noise:
        result_path += '_noise'

    timestamp = time.time()
    result_path += f'_{timestamp}.csv'
    df.to_csv(result_path)
    print(df.groupby(['# edges', 'setting']).mean())
    print(df.groupby(['# edges', 'setting']).std())
    print(f'Results are saved to {result_path}.')

def _efficiency_retrain(args, data, device):
    args.edges = [100, 200, 400, 600, 800, 1000] if args.data != 'cs' else [1000, 2000, 4000, 6000, 8000, 10000]

    result = defaultdict(list)
    for _ in tqdm(range(10), desc=f'{args.data}-{args.model}'):
        for method in ['random', 'max-degree', 'min-degree']:
            if method == 'max-degree':
                args.max_degree = True
                args.method = 'degree'
            elif method == 'min-degree':
                args.method = 'degree'
                args.max_degree = False
            else:
                args.method = method
            edges_to_forget = sample_edges(args, data, method=args.method)

            for num_edges in args.edges:
                _, retrain_time = retrain(args, data, edges_to_forget[:num_edges], device)
                result['running time'].append(retrain_time)
                result['# edges'].append(num_edges)
                result['method'].append(method)
    df = pd.DataFrame(data=result)
    print(df.groupby(['# edges', 'method']).mean())
    df.to_csv(os.path.join('./result', f'rq2_efficiency_retrain_{args.data}_{args.model}.csv'))

def _efficiency_unlearn(args, data, device):
    args.edges = [100, 200, 400, 600, 800, 1000] if args.data != 'cs' else [1000, 2000, 4000, 6000, 8000, 10000]
    original_model = train_model(args, data, eval=False, verbose=False, device=device)

    result = defaultdict(list)
    for _ in tqdm(range(10), desc=f'{args.data}-{args.model}'):
        for method in ['random', 'max-degree', 'min-degree']:
            if method == 'max-degree':
                args.max_degree = True
                args.method = 'degree'
            elif method == 'min-degree':
                args.method = 'degree'
                args.max_degree = False
            else:
                args.method = method
            edges_to_forget = sample_edges(args, data, method=args.method)

            for num_edges in args.edges:
                # _, retrain_time = retrain(args, data, edges_to_forget[:num_edges], device)
                _, unlearn_time = unlearn(args, data, original_model, edges_to_forget[:num_edges], device)
                result['running time'].append(unlearn_time)
                result['# edges'].append(num_edges)
                result['method'].append(method)
    df = pd.DataFrame(data=result)
    print(df.groupby(['# edges', 'method']).mean())
    df.to_csv(os.path.join('./result', f'rq2_efficiency_unlearn_{args.data}_{args.model}.csv'))


def _efficacy(args, data, device):
    args.edges = [100, 200, 400, 600, 800, 1000] if args.data != 'cs' else [1000, 2000, 4000, 6000, 8000, 10000]

    result = defaultdict(list)
    for _ in tqdm(range(args.num_trials), desc=f'{args.data}-{args.model}'):
        args.add_noise = False
        origin_model = train_model(args, data, eval=False, verbose=False, device=device)
        mia = train_mia(origin_model, data, device)

        args.add_noise = True
        noise_origin_model = train_model(args, data, eval=False, verbose=False, device=device)

        edges_to_forget = sample_edges(args, data, method='random')
        for num_edges in args.edges:
            _data = copy.deepcopy(data)
            _data.remove_edges(edges_to_forget[:num_edges])
            print(f'after remove {num_edges}, the number of edges is {_data.num_edges}.')
            args.add_noise = False
            # retrain_model, _ = retrain(args, _data, device)
            retrain_model = train_model(args, _data, eval=False, verbose=False, device=device)
            unlearn_model, _, = unlearn(args, origin_model.to(device), data, _data, edges_to_forget[:num_edges], device)

            args.add_noise = True
            # noise_retrain_model, _ = retrain(args, _data, device)
            noise_retrain_model = train_model(args, _data, eval=False, verbose=False, device=device)
            noise_unlearn_model, _, = unlearn(args, noise_origin_model.to(device), data, _data, edges_to_forget[:num_edges], device)


            o_result = evaluate_mia_model(mia, origin_model, data, edges_to_forget[:num_edges], device, reverse=True)
            r_mia = train_mia(retrain_model, _data, device, test_edges=edges_to_forget[:num_edges])
            r_result = evaluate_mia_model(r_mia, retrain_model, _data, edges_to_forget[:num_edges], device)
            u_mia = train_mia(unlearn_model, _data, device, test_edges=edges_to_forget[:num_edges])
            u_result = evaluate_mia_model(u_mia, unlearn_model, _data, edges_to_forget[:num_edges], device)
            
            n_o_result = evaluate_mia_model(mia, noise_origin_model, data, edges_to_forget[:num_edges], device, reverse=True)
            n_r_mia = train_mia(noise_retrain_model, _data, device, test_edges=edges_to_forget[:num_edges])
            n_r_result = evaluate_mia_model(n_r_mia, noise_retrain_model, _data, edges_to_forget[:num_edges], device)
            n_u_mia = train_mia(noise_unlearn_model, _data, device, test_edges=edges_to_forget[:num_edges])
            n_u_result = evaluate_mia_model(n_u_mia, noise_unlearn_model, _data, edges_to_forget[:num_edges], device)

            # o_result_all, o_result_class = stealing_link(origin_model, data, edges_to_forget[:num_edges], device, reverse=True)
            # r_result_all, r_result_class = stealing_link(retrain_model, _data, edges_to_forget[:num_edges], device)
            # u_result_all, u_result_class = stealing_link(unlearn_model, _data, edges_to_forget[:num_edges], device)

            result['# edges'].append(num_edges)
            result['setting'].append('Original')
            result['Stealing Link Acc'].append(o_result[0])
            result['Stealing Link Auc'].append(o_result[1])

            result['# edges'].append(num_edges)
            result['setting'].append('Retrain')
            result['Stealing Link Acc'].append(r_result[0])
            result['Stealing Link Auc'].append(r_result[1])

            result['# edges'].append(num_edges)
            result['setting'].append('UEU')
            result['Stealing Link Acc'].append(u_result[0])
            result['Stealing Link Auc'].append(u_result[1])
            
            result['# edges'].append(num_edges)
            result['setting'].append('O+N')
            result['Stealing Link Acc'].append(n_o_result[0])
            result['Stealing Link Auc'].append(n_o_result[1])

            result['# edges'].append(num_edges)
            result['setting'].append('R+N')
            result['Stealing Link Acc'].append(n_r_result[0])
            result['Stealing Link Auc'].append(n_r_result[1])

            result['# edges'].append(num_edges)
            result['setting'].append('CEU')
            result['Stealing Link Acc'].append(n_u_result[0])
            result['Stealing Link Auc'].append(n_u_result[1])

    df = pd.DataFrame(result)
    print(df.groupby(['# edges', 'setting']).mean())
    print(df.groupby(['# edges', 'setting']).std())
    df_path = os.path.join('./result', f'mia_{args.data}_{args.model}_random')
    if args.feature:
        df_path += '_feature'
    if len(args.hidden) > 0:
        df_path += f'_l{len(args.hidden)}'
    if args.add_noise:
        df_path += '_noise'
    timestamp = int(time.time())
    df_path += f'_{timestamp}.csv'
    df.to_csv(df_path)

def _vary_epsilon(args, data, device):
    epsilons = [0.1, 0.2, 0.5, 1, 5, 10]
    args.edges = [100, 200, 400, 600, 800, 1000] if args.data != 'cs' else [1000, 2000, 4000, 6000, 8000, 10000]

    result = defaultdict(list)

    for _ in tqdm(range(10), desc=f'{args.data}-{args.model}'):
        edges_to_forget = sample_edges(args, data, method=args.method)
        
        dd_bounds = []
        original = train_model(args, data, eval=False, verbose=False, device=device)
        for num_edges in args.edges:
            _data = copy.deepcopy(data)
            _data.remove_edges(edges_to_forget[:num_edges])
            unlearn_model, _, dd_bound = unlearn(args, original, data, _data, edges_to_forget[:num_edges], device, return_bound=True)
            dd_bounds.append(dd_bound) 
            
        for epsilon in tqdm(epsilons, desc='epsilon'):
            for num_edges, dd in zip(args.edges, dd_bounds):
                args.sigma = dd / epsilon
                original = train_model(args, data, eval=False, verbose=False, device=device)
                original_res = evaluate(args, data, original, device)
                result['epsilon'].append(epsilon)
                result['# edges'].append(0)
                result['setting'].append('retrain')
                result['accuracy'].append(original_res['accuracy'])
                result['dd bound'].append(dd)
                
                result['epsilon'].append(epsilon)
                result['# edges'].append(0)
                result['setting'].append('unlearn')
                result['accuracy'].append(original_res['accuracy'])
                result['dd bound'].append(dd)

                _data = copy.deepcopy(data)
                _data.remove_edges(edges_to_forget[:num_edges])

                retrain_model, _ = retrain(args, _data, device)
                retrain_res = evaluate(args, _data, retrain_model, device)
                unlearn_model, _ = unlearn(args, original, data, _data, edges_to_forget[:num_edges], device)
                unlearn_res = evaluate(args, _data, unlearn_model, device)
                
                result['epsilon'].append(epsilon)
                result['# edges'].append(num_edges)
                result['setting'].append('retrain')
                result['accuracy'].append(retrain_res['accuracy'])
                result['dd bound'].append(dd)

                result['epsilon'].append(epsilon)
                result['# edges'].append(num_edges)
                result['setting'].append('unlearn')
                result['accuracy'].append(unlearn_res['accuracy'])
                result['dd bound'].append(dd)

    df = pd.DataFrame(data=result)
    result_path = os.path.join('./result', f'epsilon_{args.data}_{args.model}_{args.method}')
    if args.feature:
        result_path += '_feature'
    if len(args.hidden) > 0:
        result_path += f'_l{len(args.hidden)}'
    if args.add_noise:
        result_path += '_noise'

    timestamp = time.time()
    result_path += f'_{timestamp}.csv'
    df.to_csv(result_path)


def _compare_cgu(args, data, device):
    unlearn_result = defaultdict(list)
    mia_result = defaultdict(list)
    args.edges = [100, 200, 400, 600, 800, 1000] if args.data != 'cs' else [1000, 2000, 4000, 6000, 8000, 10000]

    nodes = torch.tensor(data.test_set.nodes, device=device)
    for _ in tqdm(range(args.num_trials)):
        original = train_model(args, data, eval=False, verbose=False, device=device)
        original_res = evaluate(args, data, original, device)
        mia = train_mia(original, data, device)

        unlearn_result['# edges'].append(0)
        unlearn_result['setting'].append('original')
        unlearn_result['accuracy'].append(original_res['accuracy'])
        unlearn_result['f1'].append(original_res['f1'])
        unlearn_result['running time'].append(0) 

        edges_to_forget = sample_edges(args, data, method='random')
        for num_edges in args.edges:
            _data = copy.deepcopy(data)
            _data.remove_edges(edges_to_forget[:num_edges])

            retrain_model, retrain_time = retrain(args, _data, device)
            retrain_res = evaluate(args, _data, retrain_model, device)
            retrain_model.cpu()

            unlearn_model, ceu_time = unlearn(args, original.to(device), data, _data, edges_to_forget[:num_edges], device, return_bound=False)
            unlearn_res = evaluate(args, _data, unlearn_model, device)
            unlearn_model.cpu()

            torch.cuda.empty_cache()

            unlearn_result['# edges'].append(num_edges)
            unlearn_result['setting'].append('retrain')
            unlearn_result['accuracy'].append(retrain_res['accuracy'])
            unlearn_result['f1'].append(retrain_res['f1'])
            unlearn_result['running time'].append(retrain_time)
            # unlearn_result['dd bound'].append(0)

            unlearn_result['# edges'].append(num_edges)
            unlearn_result['setting'].append('ceu')
            unlearn_result['accuracy'].append(unlearn_res['accuracy'])
            unlearn_result['f1'].append(unlearn_res['f1'])
            unlearn_result['running time'].append(ceu_time)
            # unlearn_result['dd bound'].append(dd_bound)

            # unlearn_result['# edges'].append(num_edges)
            # unlearn_result['setting'].append('cgu')
            # unlearn_result['accuracy'].append(cgu_res['accuracy'])
            # unlearn_result['f1'].append(cgu_res['f1'])
            # unlearn_result['running time'].append(cgu_time)
            # unlearn_result['dd bound'].append(cgu_bound)
            
            o_result = evaluate_mia_model(mia, original, data, edges_to_forget[:num_edges], device, reverse=True)
            r_result = evaluate_mia_model(mia, retrain_model, _data, edges_to_forget[:num_edges], device)
            u_result = evaluate_mia_model(mia, unlearn_model, _data, edges_to_forget[:num_edges], device)
            # c_result = evaluate_mia_model(mia, cgu_model, _data, edges_to_forget[:num_edges], device)
            torch.cuda.empty_cache()

            mia_result['# edges'].append(num_edges)
            mia_result['setting'].append('Original')
            mia_result['acc'].append(o_result[0])
            mia_result['auc'].append(o_result[1])
            mia_result['# edges'].append(num_edges)
            mia_result['setting'].append('Retrain')
            mia_result['acc'].append(r_result[0])
            mia_result['auc'].append(r_result[1])
            # mia_result['# edges'].append(num_edges)
            # mia_result['setting'].append('cgu')
            # mia_result['acc'].append(c_result[0])
            # mia_result['auc'].append(c_result[1])
            mia_result['# edges'].append(num_edges)
            mia_result['setting'].append('unlearn')
            mia_result['acc'].append(u_result[0])
            mia_result['auc'].append(u_result[1])

        cgu_res = sgc_edge_unlearn(args, original, data, edges_to_forget[:args.edges[-1]], device, mia=mia)
        unlearn_result['# edges'].extend(cgu_res['# edges'])
        unlearn_result['setting'].extend(['cgu'] * len(cgu_res['# edges']))
        unlearn_result['accuracy'].extend(cgu_res['accuracy'])
        unlearn_result['f1'].extend(cgu_res['f1'])
        unlearn_result['running time'].extend(cgu_res['running time'])

        mia_result['# edges'].extend(cgu_res['# edges'])
        mia_result['setting'].extend(['cgu'] * len(cgu_res['# edges']))
        mia_result['acc'].extend(cgu_res['mia acc'])
        mia_result['auc'].extend(cgu_res['mia auc'])


    unlearn_df = pd.DataFrame(unlearn_result)
    mia_df = pd.DataFrame(data=mia_result)
    # print(unlearn_df.groupby(['# edges', 'setting']).mean())
    timestamp = int(time.time())
    unlearn_result_path = os.path.join('./result', f'certified_unlearn_{args.data}_{args.model}_{args.method}')
    mia_result_path = os.path.join('./result', f'certified_mia_{args.data}_{args.model}_{args.method}')
    if args.feature:
        unlearn_result_path += '_feature'
        mia_result_path += '_feature'
    if len(args.hidden) > 0:
        unlearn_result_path += f'_l{len(args.hidden)}'
        mia_result_path += f'_l{len(args.hidden)}'
    if args.add_noise:
        unlearn_result_path += f'_noise'
        mia_result_path += '_noise'

    unlearn_result_path += f'_{timestamp}.csv'
    mia_result_path += f'_{timestamp}.csv'
    unlearn_df.to_csv(unlearn_result_path)
    mia_df.to_csv(mia_result_path)



if __name__ == '__main__':
    parser = argument_parser()
    parser.add_argument('-rq', type=str, default=None)
    parser.add_argument('-analysis', type=str, default=None)
    parser.add_argument('-mia', dest='mia_attack', action='store_true',
                        help='Indicator of evaluting the unlearning model via MIA attack (accuracy).')
    parser.add_argument('-cosine', dest='cosine', action='store_true')
    parser.add_argument('-datasets', type=str, nargs='+', default=['cora', 'citeseer'])
    parser.add_argument('-targets', type=str, nargs='+', default=['gcn', 'sage', 'gin'])

    args = parser.parse_args()
    print('Parameters:', vars(args))
    device = torch.device(f'cuda:{args.gpu}') if torch.cuda.is_available() and args.gpu >= 0 else torch.device('cpu')

    for d in args.datasets:
        args.data = d
        data = load_data(args)
        for target in args.targets:
            args.model = target
            if args.rq == 'bound':
                _gradient_residual(args, data, device)
            elif args.rq == 'unlearn':
                _unlearning(args, data, device)
            elif args.rq == 'efficiency':
                _efficiency_retrain(args, data, device)
                _efficiency_unlearn(args, data, device)
            elif args.rq == 'efficacy':
                _efficacy(args, data, device)
            elif args.rq == 'epsilon':
                _vary_epsilon(args, data, device)
            elif args.rq == 'cgu_compare':
                _compare_cgu(args, data, device) 
