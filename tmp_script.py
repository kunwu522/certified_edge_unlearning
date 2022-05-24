import os
import pandas as pd
import numpy as np


if __name__ == '__main__':
    # for d in ['cora', 'citeseer', 'polblogs']:
    #     for m in ['gcn', 'gat', 'sage', 'gin']:
    #         df = pd.read_csv(os.path.join('./result', f'rq1_unlearn_{d}_{m}_l1_16.csv'))
    #         df['# edges'] = [200, 400, 600, 800, 1000, 1200, 1400, 1600, 1800, 2000] * 10
    #         _df = df[(df['# edges'] == 200) | (df['# edges'] == 400) | (df['# edges'] == 800) | (df['# edges'] == 1000)]
    #         pd.DataFrame(data={
    #             '# edges': _df['# edges'].values.tolist() * 2,
    #             'accuracy': _df['retrain-acc'].values.tolist() + _df['unlearn-acc'].values.tolist(),
    #             'setting': ['Retrain'] * 40 + ['ERAEDGE'] * 40}).to_csv(f'./result/_rq1_fidelity_{d}_{m}.csv')

    for d in ['cora', 'citeseer']:
        for m in ['gcn', 'gat', 'sage', 'gin']:
            print(m, d)
            df = pd.read_csv(os.path.join('./result', f'rq1_baseline_{d}_{m}_l1_16.csv'), index_col=0)
            _df = df[(df['# edges'] == 200) | (df['# edges'] == 400) | (df['# edges'] == 800) | (df['# edges'] == 1000)]
            _df['setting'] = ['BLPA', 'BEKM'] * int(len(_df) / 2)
            _df.to_csv(f'./result/_rq1_fidelity_baseline_{d}_{m}.csv')

            baseline_df = pd.read_csv(os.path.join('./result', f'rq1_fidelity_baseline_{d}_{m}.csv'))
            baseline_df['setting'] = ['BLPA', 'BEKM'] * int(len(baseline_df) / 2)
            baseline_df.to_csv(os.path.join('./result', f'rq1_fidelity_baseline_{d}_{m}.csv'))
