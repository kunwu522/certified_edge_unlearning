from collections import defaultdict
from email.policy import default
import os
import argparse
from turtle import color
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import seaborn as sns


dataset_num_edges = {
    'cora': 5429,
    'citeseer': 9104,
    'polblogs': 19025,
    'physics': 495924,
}

dataset_num_nodes = {
    'cora': 1732,
    'citeseer': 2128,
    'polblogs': 19025,
    'physics': 495924,
}


def _approximation_evaluate(args):
    # df_list = []
    # for d in ['cora', 'citeseer']:
    #     for m in ['gcn', 'gat', 'sage']:
    #         df = pd.read_csv(os.path.join('./result', f'appr_loss_{d}_{m}.csv'))
    #         df_list.append(df)
    # df = pd.concat(df_list, ignore_index=True)
    df = pd.read_csv(os.path.join('./result', f'appr_loss_{args.data}_{args.model}.csv'))

    sns.set_theme(style='whitegrid')
    plt.rc('axes', labelsize=20)
    plt.rc('xtick', labelsize=18)
    plt.rc('ytick', labelsize=18)
    plt.rc('font', size=20)
    ax = sns.scatterplot(data=df, x='retrain_loss', y='unlearn_loss', s=80)
    ax.set_xlabel('Loss (retrain)')
    ax.set_ylabel('Loss (unlearn)')
    # plt.xlim([0.1, 0.2])
    # plt.ylim([0.15, 0.2])
    # plt.legend()
    plt.subplots_adjust(left=0.17, bottom=0.14)
    plt.savefig(os.path.join('./plot', f'loss_{args.data}_{args.model}.pdf'), bbox_inches='tight')
    plt.show()


def RQ4_adversarial_vs_benign(args):
    if args.hidden:
        df = pd.read_csv(os.path.join(
            './result', f'rq4_diff_{args.data}_{args.model}_h{len(args.hidden)}_{"_".join(map(str, args.hidden))}.csv'))
    else:
        df = pd.read_csv(os.path.join('./result', f'rq4_diff_{args.data}_{args.model}.csv'))

    sns.set_theme(style='darkgrid')
    plt.figure(figsize=(10, 6))
    plt.rc('axes', labelsize=20)
    plt.rc('xtick', labelsize=18)
    plt.rc('ytick', labelsize=18)
    plt.rc('legend', fontsize=16)
    plt.rc('font', size=20)
    ax = sns.barplot(x='# edges', y='Model Accuracy', hue='type', data=df, palette=sns.color_palette("Set2", 3))
    ax.set_xticklabels([f'{int(x / dataset_num_edges[args.data] * 100)}%' for x in df['# edges'].unique()])
    ax.set_xlabel('% of unlearned edges')
    ax.set_ylabel('Accuracy Difference')
    # plt.legend(loc='upper left', ncol=3)
    plt.legend()
    plt.subplots_adjust(bottom=0.12)
    if args.hidden:
        figure_filename = f'rq4_diff_{args.data}_{args.model}_h{len(args.hidden)}_{"_".join(map(str, args.hidden))}.pdf'
    else:
        figure_filename = f'rq4_diff_{args.data}_{args.model}.pdf'
    plt.savefig(os.path.join('./plot', figure_filename), bbox_inches='tight')
    plt.show()


def _rq2_efficiency(args):
    unlearn_df = pd.read_csv(os.path.join('./result', f'rq2_efficiency_unlearn_{args.data}_{args.model}.csv'))
    unlearn_df = unlearn_df[~unlearn_df['method'].str.startswith('saliency')]
    unlearn_df['method'] = unlearn_df['method'].str.replace('random', 'EraEdge(Rand)')
    unlearn_df['method'] = unlearn_df['method'].str.replace('max-degree', 'EraEdge(MaxD)')
    unlearn_df['method'] = unlearn_df['method'].str.replace('min-degree', 'EraEdge(MinD)')

    retrain_df = pd.read_csv(os.path.join('./result', f'rq2_efficiency_retrain_{args.data}_{args.model}.csv'))
    retrain_df = retrain_df[~retrain_df['method'].str.startswith('saliency')]
    retrain_df['method'] = retrain_df['method'].str.replace('random', 'Retrain(Rand)')
    retrain_df['method'] = retrain_df['method'].str.replace('max-degree', 'Retrain(MaxD)')
    retrain_df['method'] = retrain_df['method'].str.replace('min-degree', 'Retrain(MinD)')

    df = pd.concat((retrain_df, unlearn_df), axis=0, ignore_index=True)

    sns.set_theme(style='darkgrid')
    plt.figure(figsize=(8, 6))
    plt.rc('axes', labelsize=20)
    plt.rc('xtick', labelsize=20)
    plt.rc('ytick', labelsize=20)
    plt.rc('legend', fontsize=16)
    plt.rc('font', size=20)
    c = sns.color_palette()
    ax = sns.barplot(data=df, x='# edges', y='running time', hue='method',
                     hue_order=['Retrain(Rand)', 'EraEdge(Rand)', 'Retrain(MaxD)', 'EraEdge(MaxD)', 'Retrain(MinD)', 'EraEdge(MinD)'])
    labels = ax.get_legend_handles_labels()

    ax.set_xlabel('Number of unlearned edges')
    ax.set_ylabel('Unlearning time (seconds)')
    # ax.set_xticklabels([f'{int(x / dataset_num_edges[args.data] * 100)}%' for x in df['# edges'].unique()])
    # ax.set_xticks([100, 200, 400, 800, 1000])
    plt.legend([], [], frameon=False)
    plt.subplots_adjust(bottom=0.15)
    plt.savefig(os.path.join('./plot/', f'rq2_efficiency_{args.data}_{args.model}.pdf'), bbox_inches='tight')
    plt.show()

    fig_legend, ax_legend = plt.subplots(figsize=(10, 1))
    # ax.legend
    ax_legend.axis(False)
    ax_legend.legend(*labels, loc='center', ncol=3)
    fig_legend.savefig(os.path.join('./plot', f'rq2_effiency_legend.pdf'), bbox_inches='tight')


def rq1_efficiency(args):
    unlearn_df = pd.read_csv(os.path.join('./result', f'rq1_efficiency_unlearn_{args.data}_{args.model}.csv'))
    retrain_df = pd.read_csv(os.path.join('./result', f'rq1_efficiency_retrain_{args.data}_{args.model}_l1.csv'))
    if args.data == 'cs' or args.data == 'physics':
        baseline_df = pd.read_csv(os.path.join('./result', f'rq1_baseline_efficiency_{args.data}_{args.model}.csv'))
        print(baseline_df)
        # baseline_df = pd.DataFrame(np.repeat(_df.values, 5, axis=0))
        # baseline_df.columns = _df.columns
        # baseline_df['# edges'] = [100, 200, 400, 800, 1000] * 20
    else:
        baseline_df = pd.read_csv(os.path.join('./result', f'rq1_fidelity_baseline_{args.data}_{args.model}.csv'))
        baseline_df = baseline_df[baseline_df['# edges'] != 0]
    # baseline_df = baseline_df[_baseline_df['# edges'].isin([100, 200, 400, 800, 1000])]
    # for m in ['gcn', 'gat', 'sage', 'gin']:
    #     _baseline_df = pd.read_csv(os.path.join('./result', f'rq1_fidelity_baseline_{args.data}_{m}.csv'))
    #     _baseline_df = _baseline_df[_baseline_df['# edges'].isin([100, 200, 400, 800, 1000])]
    #     _baseline_df = [f'{p}({m})' for p in _baseline_df['partition'].values]
    #     baseline_df.append(_baseline_df)
    # baseline_df = pd.concat(baseline_df, ignore_index=True)

    df = pd.DataFrame()
    df['# edges'] = pd.concat((baseline_df['# edges'], retrain_df['# edges'], retrain_df['# edges']), ignore_index=True)
    df['running time'] = pd.concat((baseline_df['running time'] / baseline_df['# shards'], retrain_df['running time'],
                                   unlearn_df['running time']), ignore_index=True)
    df['setting'] = baseline_df['partition'].str.upper().values.tolist() + ['Retrain'] * \
        retrain_df.shape[0] + ['EraEdge'] * unlearn_df.shape[0]

    sns.set_theme(style='darkgrid')
    plt.figure(figsize=(10, 6))
    plt.rc('axes', labelsize=20)
    plt.rc('xtick', labelsize=18)
    plt.rc('ytick', labelsize=18)
    plt.rc('legend', fontsize=16)
    plt.rc('font', size=20)
    c = sns.color_palette()
    # ax = sns.lineplot(data=df, x='# edges', y='running time', hue='setting',
    #                   style='setting', markers=False, linewidth=2.5,
    #                   palette=[c[0], 'black', c[2], c[3]])
    ax = sns.barplot(data=df, x='# edges', y='running time', hue='setting',
                     palette=[c[0], 'black', c[2], c[3]])
    labels = ax.get_legend_handles_labels()
    print(labels)

    ax.set_xlabel('Number of unlearned edges')
    ax.set_ylabel('Unlearning time (seconds)')
    # ax.set_xticklabels([f'{int(x / dataset_num_edges[args.data] * 100)}%' for x in df['# edges'].unique()])
    # ax.set_xticks([100, 200, 400, 800, 1000])
    plt.legend([], [], frameon=False)
    plt.subplots_adjust(bottom=0.15)
    plt.savefig(os.path.join('./plot/', f'rq1_efficiency_{args.data}_{args.model}.pdf'), bbox_inches='tight')
    plt.show()

    fig_legend, ax_legend = plt.subplots(figsize=(10, 1))
    # ax.legend
    ax_legend.axis(False)
    ax_legend.legend(*labels, loc='center', ncol=4)
    fig_legend.savefig(os.path.join('./plot', f'rq1_effiency_legend.pdf'), bbox_inches='tight')


def RQ4_adversarial_edges_unlearn(args):
    if args.hidden:
        df = pd.read_csv(os.path.join(
            './result', f'rq4_unlearn_{args.data}_{args.model}_h{len(args.hidden)}_{"_".join(map(str, args.hidden))}.csv'))
    else:
        df = pd.read_csv(os.path.join('./result', f'rq4_unlearn_{args.data}_{args.model}.csv'))
    sns.set_theme(style='darkgrid')
    plt.figure(figsize=(10, 6))
    plt.rc('axes', labelsize=20)
    plt.rc('xtick', labelsize=16)
    plt.rc('ytick', labelsize=16)
    plt.rc('legend', fontsize=16)
    plt.rc('font', size=20)

    sns.lineplot(x='# edges', y='Accuracy', data=df, hue='setting',
                 linewidth=2.5, style='setting', markers=True,
                 palette=sns.color_palette('Set2', 3),
                 markersize=13)

    plt.xticks(df['# edges'].unique(), [
               f'{int(x / dataset_num_edges[args.data] * 100)}%' for x in df['# edges'].unique()])
    plt.xlabel('% of adversrial edges')
    plt.ylabel('Model accuracy')
    plt.legend()
    plt.subplots_adjust(bottom=0.15)
    if args.hidden:
        figure_filename = f'rq4_unlearn_{args.data}_{args.model}_h{len(args.hidden)}_{"_".join(map(str, args.hidden))}.pdf'
    else:
        figure_filename = f'rq4_unlearn_{args.data}_{args.model}.pdf'
    plt.savefig(os.path.join('./plot', figure_filename), dpi=400, bbox_inches='tight')
    plt.show()


def _rq2_fidelity(args):
    df = pd.read_csv(os.path.join('./result', f'rq2_fidelity_{args.data}_{args.model}.csv'))
    df = df[~df.setting.str.endswith('O')]
    df = df[~df.setting.str.startswith('saliency')]
    df['# edges'] = df['# edges'].values.astype(str)
    sns.set_theme(style='darkgrid')
    plt.figure(figsize=(8, 6))
    plt.rc('axes', labelsize=20)
    plt.rc('xtick', labelsize=20)
    plt.rc('ytick', labelsize=20)
    plt.rc('legend', fontsize=16)
    plt.rc('font', size=20)

    c = sns.color_palette('Paired', 8)
    ax = sns.lineplot(x='# edges', y='accuracy', data=df, hue='setting',
                      linewidth=2.5, style='setting',
                      dashes=['', (4, 2), '', (4, 2), '', (4, 2)],
                      palette=[c[1], c[1], c[3], c[3], c[5], c[5]],
                      markersize=13, ci=None)
    labels = plt.gca().get_legend_handles_labels()

    # ax.set_xticklabels(list(range(200, 2200, 200)))
    # ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    # ax.set_xticks([100, 200, 400, 800, 1000])
    # ax.set_ylim([0.83, 0.88])
    # plt.xticklabels([f'{int(x/dataset_num_edges[args.data] * 100)}%' for x in df['# edges']])
    plt.xlabel('Number of unlearned edge')
    plt.ylabel('Model accuracy')
    plt.legend([], [], frameon=False)
    plt.subplots_adjust(bottom=0.15, left=0.15)

    figure_filename = f'rq2_fidelity_{args.data}_{args.model}.pdf'
    plt.savefig(os.path.join('./plot', figure_filename), bbox_inches='tight')
    plt.show()

    fig_legend, ax_legend = plt.subplots(figsize=(10, 1))
    # ax.legend
    ax_legend.axis(False)
    legend_names = [
        'Retrain(Rand)', 'ERAEDGE(Rand)',
        'Retrain(MaxD)', 'EraEdge(MaxD)',
        'Retrain(MinD)', 'EraEdge(MinD)',
        # 'Retrain(Saliency@K)', 'ERAEDGE(Saliency@K)',
    ]
    ax_legend.legend(labels[0], legend_names, loc='center', ncol=3)

    fig_legend.savefig(os.path.join('./plot', f'rq2_edge_sampling_legend.pdf'), bbox_inches='tight')


def _edges_sampling(args):
    # if args.hidden:
    #     df = pd.read_csv(os.path.join(
    #         './result', f'rq2_sample_{args.data}_{args.model}_h{len(args.hidden)}_{"_".join(map(str, args.hidden))}.csv'), index_col=0)
    # else:
    #     df = pd.read_csv(os.path.join('./result', f'rq2_sample_{args.data}_{args.model}.csv'))
    df = pd.read_csv(os.path.join(
        './result', f'rq2_{args.data}_{args.model}.csv'))

    print(df.head())

    df = df[~df.setting.str.endswith('O')]

    sns.set_theme(style='darkgrid')
    plt.figure(figsize=(10, 6))
    plt.rc('axes', labelsize=20)
    plt.rc('xtick', labelsize=16)
    plt.rc('ytick', labelsize=16)
    plt.rc('legend', fontsize=16)
    plt.rc('font', size=20)

    c = sns.color_palette('Paired', 8)
    ax = sns.lineplot(x='# edges', y='accuracy', data=df, hue='setting',
                      linewidth=2.5, style='setting',
                      dashes=['', (5, 2), '', (5, 2), '', (5, 2), '', (5, 2)],
                      palette=[c[1], c[1], c[3], c[3], c[5], c[5], c[7], c[7]],
                      markersize=13, ci=None)
    # sns.lineplot(x='# edges', y='random-R', data=df, palette=c[0], label='Random-R', linewidth=2.5, markers=True)
    # sns.lineplot(x='# edges', y='random-U', data=df, palette=c[1], label='Random-U', linewidth=2.5)
    # sns.lineplot(x='# edges', y='max-degree-R', data=df, palette=c[2], label='MaxDegree-R', linewidth=2.5)
    # sns.lineplot(x='# edges', y='max-degree-U', data=df, palette=c[3], label='MaxDegree-U', linewidth=2.5)
    # sns.lineplot(x='# edges', y='min-degree-R', data=df, palette=c[4], label='MinDegree-R', linewidth=2.5)
    # sns.lineplot(x='# edges', y='min-degree-U', data=df, palette=c[5], label='MinDegree-U', linewidth=2.5)
    # sns.lineplot(x='# edges', y='saliency-R', data=df, palette=c[6], label='Saliency-R', linewidth=2.5)
    # sns.lineplot(x='# edges', y='saliency-U', data=df, palette=c[7], label='Saliency-U', linewidth=2.5)

    # plt.plot(df[df['setting'] == 'random-R']['# edges'], df[df['setting'] == 'random-R']
    #          ['Accuracy'], '-', label='Retrain(Random@K)', linewidth=2.5, color=c[1])
    # plt.plot(df[df['setting'] == 'random-R']['# edges'], df[df['setting'] == 'random-U']
    #          ['Accuracy'], ':', label='ERAEDGE(Random@K)', linewidth=2.5, color=c[1])
    # plt.plot(df[df['setting'] == 'max-degree-R']['# edges'], df[df['setting'] == 'max-degree-R']
    #          ['Accuracy'], '-', label='Retrain(MaxDegree@K)', linewidth=2.5, color=c[3])
    # plt.plot(df[df['setting'] == 'max-degree-R']['# edges'], df[df['setting'] == 'max-degree-U']
    #          ['Accuracy'], ':', label='ERAEDGE(MaxDegree@K)', linewidth=2.5, color=c[3])
    # plt.plot(df[df['setting'] == 'min-degree-R']['# edges'], df[df['setting'] == 'min-degree-R']
    #          ['Accuracy'], '-', label='Retrain(MaxDegree@K)', linewidth=2.5, color=c[5])
    # plt.plot(df[df['setting'] == 'min-degree-R']['# edges'], df[df['setting'] == 'min-degree-U']
    #          ['Accuracy'], ':', label='ERAEDGE(MaxDegree@K)', linewidth=2.5, color=c[5])
    # plt.plot(df[df['setting'] == 'saliency-R']['# edges'], df[df['setting'] == 'saliency-R']
    #          ['Accuracy'], '-', label='Retrain(Saliency@K)', linewidth=2.5, color=c[7])
    # plt.plot(df[df['setting'] == 'saliency-R']['# edges'], df[df['setting'] == 'saliency-U']
    #          ['Accuracy'], ':', label='ERAEDGE(Saliency@K)', linewidth=2.5, color=c[7])

    labels = plt.gca().get_legend_handles_labels()

    # plt.xticks(df['# edges'])
    ax.set_xticklabels(list(range(200, 2200, 200)))
    # plt.xticklabels([f'{int(x/dataset_num_edges[args.data] * 100)}%' for x in df['# edges']])
    plt.xlabel('Number of unlearned edge')
    plt.ylabel('Model accuracy')
    plt.legend([], [], frameon=False)
    plt.subplots_adjust(bottom=0.15)

    figure_filename = f'edge_sampling_{args.data}_{args.model}.pdf'
    plt.savefig(os.path.join('./plot', figure_filename), bbox_inches='tight')
    plt.show()

    fig_legend, ax_legend = plt.subplots(figsize=(10, 1))
    # ax.legend
    ax_legend.axis(False)
    legend_names = [
        'Retrain(Rand@K)', 'ERAEDGE(Rand@K)',
        'Retrain(MaxDegree@K)', 'ERAEDGE(MaxDegree@K)',
        'Retrain(MinDegree@K)', 'ERAEDGE(MinDegree@K)',
        'Retrain(Saliency@K)', 'ERAEDGE(Saliency@K)',
    ]
    ax_legend.legend(labels[0], legend_names, loc='center', ncol=4)

    fig_legend.savefig(os.path.join('./plot', f'rq2_edge_sampling_legend.pdf'), bbox_inches='tight')


def _rq4_fidelity(args):
    df = []
    for m in ['gcn', 'gat', 'sage', 'gin']:
        _df = pd.read_csv(os.path.join('./result', f'rq2_{args.data}_{m}_l6.csv'))
        _df = _df[(_df['Setting'] == 'RI-retrain') | (_df['Setting'] == 'RI-ours')]
        _df['model'] = [f'Retrain({m.upper()})', f'ERAEDGE({m.upper()})'] * 50
        df.append(_df)
    df = pd.concat(df, axis=0, ignore_index=True)
    # df = pd.read_csv(os.path.join('./result', f'rq2_{args.data}_{args.model}_l6.csv'))
    # df = df[df['Setting'] != 'RI-original']
    # df = df[df['Setting'] != 'NF-original']

    sns.set_theme(style='whitegrid')
    plt.figure(figsize=(10, 6))
    plt.rc('axes', labelsize=20)
    plt.rc('xtick', labelsize=18)
    plt.rc('ytick', labelsize=18)
    plt.rc('legend', fontsize=16)
    plt.rc('font', size=20)
    c = sns.color_palette()
    ax = sns.barplot(x='# Layer', y='Accuracy', hue='model', data=df,
                     palette=[c[0], c[0], c[1], c[1], c[2], c[2], c[3], c[3]])
    labels = ax.get_legend_handles_labels()

    ax.set_xlabel('Number of Layers')
    ax.set_xticklabels(list(range(2, 7)))
    ax.set_ylabel('Model accuracy')
    # plt.ylim(0.6, 0.9)

    plt.legend([], [], frameon=False)
    plt.subplots_adjust(bottom=0.12)
    plt.savefig(os.path.join('./plot', f'./rq4_fidelity_{args.data}_{args.model}_l6.pdf'))
    plt.show()

    # labels = ax.get_legend_handles_labels()

    # print(labels)

    fig_legend, ax_legend = plt.subplots(figsize=(10, 1))
    # ax.legend
    ax_legend.axis(False)
    ax_legend.legend(*labels, loc='center', ncol=4)
    fig_legend.savefig(os.path.join('./plot', f'rq4_fidelity_legend.pdf'), bbox_inches='tight')


def RQ2_legend(args):
    df = pd.read_csv(os.path.join('./result', f'rq2_{args.model}_{args.data}.csv'))
    df = df[df['Setting'] != 'RI-original']
    df = df[df['Setting'] != 'NF-original']

    sns.set_style('whitegrid')
    plt.rc('legend', fontsize=20)

    fig = plt.figure("Line plot")
    legendFig = plt.figure("Legend plot", figsize=(6, 1))

    ax = sns.barplot(x='# Layer', y='Accuracy', hue='Setting', data=df)

    legendFig.legend(
        labels=[
            "Retrain(SO)", "Ours(SO)", "Retrain", "Ours",
        ],
        loc='center', ncol=4)
    legendFig.savefig(os.path.join('./plot', 'rq2_legend.pdf'))


def node_unlearn(args):
    df = pd.read_csv(os.path.join('./result', f'node_unlearn_{args.model}_{args.data}.csv'), index_col=0)
    x = df[df['setting'] == 'RE']['# edges']
    retrain_acc = df[df['setting'] == 'RE']['accuracy']
    retrain_time = df[df['setting'] == 'RE']['time']
    unlearn_acc = df[df['setting'] == 'UL']['accuracy']
    unlearn_time = df[df['setting'] == 'UL']['time']
    # blpa_acc = df['blpa-acc']
    # blpa_time = df['blpa-time'] / df['blpa-#-shards']
    # bekm_acc = df['bekm-acc']
    # bekm_time = df['bekm-time'] / df['bekm-#-shards']
    # unlearn_acc = df['unlearn-acc']
    # unlearn_time = df['unlearn-time']

    sns.set_style('whitegrid')
    plt.rc('axes', labelsize=22)
    plt.rc('xtick', labelsize=24)
    plt.rc('ytick', labelsize=24)
    # plt.rc('legend', fontsize=16)
    plt.rc('font', size=20)

    plt.figure(figsize=(10, 10))
    ax = plt.axes(projection='3d')
    ax.xaxis.labelpad = 12
    ax.yaxis.labelpad = 12
    ax.zaxis.labelpad = 12
    ax.plot(x, retrain_acc, retrain_time, '*-', label='Retrain', linewidth=2.5, markersize=13)
    # ax.plot(df.index, blpa_acc, blpa_time, '^--', label='BLPA', linewidth=2.5, markersize=13)
    # ax.plot(df.index, bekm_acc, bekm_time, 'v-.', label='BEKM', linewidth=2.5, markersize=13)
    ax.plot(x, unlearn_acc, unlearn_time, 'o:', label='ours', linewidth=2.5, markersize=13)

    ax.set_xticks([20, 60, 100, 140, 180])
    ax.set_xticklabels([f'{int(x/dataset_num_nodes[args.data] * 100)}%' for x in [20, 60, 100, 140, 180]])
    ax.set_xlabel('% of nodes', linespacing=3.4)
    ax.set_ylabel('Accuracy', linespacing=3.2)
    ax.set_zlabel('Running time (s)', linespacing=3.2)
    # ax.set(xticks=df.index, xlabel='Number of edges to forget', ylabel='Accuracy')
    # ax.legend()
    # plt.subplots_adjust(top=-0.1)
    # plt.tight_layout()
    ax.view_init(elev=20, azim=-40)
    plt.savefig(os.path.join('./plot', f'node_unlearn_{args.model}_{args.data}.pdf'), dpi=400, bbox_inches='tight')
    plt.show()


def _unlearn_running_time(args):
    data = defaultdict(list)
    for target in ['gcn', 'gat', 'sage', 'gin']:
        if args.hidden:
            unlearn_df = pd.read_csv(os.path.join(
                './result', f'rq1_unlearn_{args.data}_{target}_l{len(args.hidden)}_{"_".join(map(str, args.hidden))}.csv'), index_col=0)
        else:
            unlearn_df = pd.read_csv(os.path.join('./result', f'rq1_unlearn_{args.data}_{target}.csv'), index_col=0)

        data['# edges'].extend(list(range(200, 2200, 200)) * 10)
        data['Setting'].extend([f'Retrain({target.upper()})' if target != 'sage' else 'Retrain(GraphSAGE)'] * 100)
        data['Unlearning time (seconds)'].extend(unlearn_df['retrain-time'].values.tolist())
        data['# edges'].extend(list(range(200, 2200, 200)) * 10)
        data['Setting'].extend([f'ERAEDGE({target.upper()})' if target != 'sage' else 'ERAEDGE(GraphSAGE)'] * 100)
        data['Unlearning time (seconds)'].extend(unlearn_df['unlearn-time'].values.tolist())

    df = pd.DataFrame(data=data)
    sns.set_style('darkgrid')
    plt.figure(figsize=(10, 6))
    plt.rc('axes', labelsize=20)
    plt.rc('xtick', labelsize=18)
    plt.rc('ytick', labelsize=18)
    plt.rc('legend', fontsize=16)
    plt.rc('font', size=20)
    c = sns.color_palette()
    ax = sns.lineplot(data=df, x='# edges', y='Unlearning time (seconds)', hue='Setting',
                      style='Setting', markers=False, linewidth=2.5, markersize=13, ci=None,
                      palette=[c[0], c[0], c[1], c[1], c[2], c[2], c[3], c[3]],
                      dashes=['', (5, 2), '', (5, 2), '', (5, 2), '', (5, 2)])
    labels = ax.get_legend_handles_labels()

    ax.set_xlabel('Number of unlearned edges')
    # ax.set_xticklabels([f'{int(x / dataset_num_edges[args.data] * 100)}%' for x in df['# edges'].unique()])
    ax.set_xticklabels(list(range(200, 2200, 200)))
    plt.legend([], [], frameon=False)
    if args.hidden:
        figure_filename = f'rq1_time_{args.data}_l{len(args.hidden)}_{"_".join(map(str, args.hidden))}.pdf'
    else:
        figure_filename = f'rq1_time_{args.data}.pdf'
    plt.savefig(os.path.join('./plot/', figure_filename), bbox_inches='tight')
    plt.show()

    fig_legend, ax_legend = plt.subplots(figsize=(10, 1))
    # ax.legend
    ax_legend.axis(False)
    ax_legend.legend(*labels, loc='center', ncol=4)
    fig_legend.savefig(os.path.join('./plot', f'running_time_legend.pdf'), bbox_inches='tight')


def RQ1_running_time(args):
    if args.hidden:
        unlearn_df = pd.read_csv(os.path.join(
            './result', f'rq1_unlearn_{args.data}_{args.model}_l{len(args.hidden)}_{"_".join(map(str, args.hidden))}.csv'), index_col=0)
        # baseline_df = pd.read_csv(os.path.join(
        #     './result', f'rq1_baseline_{args.data}_{args.model}_h{len(args.hidden)}_{"_".join(map(str, args.hidden))}.csv'), index_col=0)
    else:
        unlearn_df = pd.read_csv(os.path.join('./result', f'rq1_unlearn_{args.data}_{args.model}.csv'), index_col=0)
        baseline_df = pd.read_csv(os.path.join('./result', f'rq1_retrain_{args.data}_{args.model}.csv'), index_col=0)

    df = pd.DataFrame({
        '# edges': list(range(200, 2200, 200)) * 20,
        'Setting': ['Retrain'] * 100 + ['ERAEDGE'] * 100,
        'Unlearning time (seconds)': np.concatenate((unlearn_df['retrain-time'].values, unlearn_df['unlearn-time'].values)),
    })

    sns.set_style('darkgrid')
    plt.figure(figsize=(10, 6))
    plt.rc('axes', labelsize=20)
    plt.rc('xtick', labelsize=18)
    plt.rc('ytick', labelsize=18)
    plt.rc('legend', fontsize=16)
    plt.rc('font', size=20)
    ax = sns.lineplot(data=df, x='# edges', y='Unlearning time (seconds)', hue='Setting',
                      style='Setting', markers=True, linewidth=2.5, markersize=13)
    ax.set_xlabel('% of unlearned edges')
    ax.set_xticklabels([f'{int(x / dataset_num_edges[args.data] * 100)}%' for x in df['# edges'].unique()])
    plt.legend()
    if args.hidden:
        figure_filename = f'rq1_time_{args.data}_{args.model}_l{len(args.hidden)}_{"_".join(map(str, args.hidden))}.pdf'
    else:
        figure_filename = f'rq1_time_{args.data}_{args.model}.pdf'
    plt.savefig(os.path.join('./plot/', figure_filename), bbox_inches='tight')
    plt.show()


def _rq3_jsd(args):
    df = pd.read_csv(f'./result/rq3_jsd_{args.data}_{args.model}.csv')
    df['# edges'] = df['# edges'].values.astype(str)

    sns.set_style('darkgrid')
    plt.figure(figsize=(10, 6))
    plt.rc('axes', labelsize=20)
    plt.rc('xtick', labelsize=18)
    plt.rc('ytick', labelsize=18)
    plt.rc('legend', fontsize=16)
    plt.rc('font', size=20)
    ax = sns.lineplot(data=df, x='# edges', y='jsd', hue='setting',
                      style='setting', markers=True, linewidth=2.5, markersize=13, ci=None)

    labels = ax.get_legend_handles_labels()

    ax.set_ylabel('JSD')
    ax.set_xlabel('Number of unlearned edges')
    # ax.set_xticklabels([f'{int(x / dataset_num_edges[args.data] * 100)}%' for x in df['# edges'].unique()])
    # ax.set_xticklabels(list(range(200, 2200, 200)))
    # ax.set_ylim(0, 0.1)
    plt.legend([], [], frameon=False)
    plt.savefig(os.path.join('./plot/', f'rq3_jsd_{args.data}_{args.model}.pdf'), bbox_inches='tight')
    plt.show()

    fig_legend, ax_legend = plt.subplots(figsize=(10, 1))
    ax_legend.axis(False)
    ax_legend.legend(labels[0], ['Adversarial', 'Benign'], loc='center', ncol=4)
    fig_legend.savefig(os.path.join('plot', 'rq3_jsd_legend.pdf'), bbox_inches='tight')


def _rq2_jsd(args):
    # df = []
    # for m in ['gcn']:
    df = pd.read_csv(os.path.join('./result', f'rq2_jsd_{args.data}_{args.model}.csv'))
    df['method'] = [f'Rand', f'MaxD', f'MinD'] * 50
    #  f'MinDegree@K({m.upper()})', f'Saliency@K({m.upper()})'] * 50
    # df.append(_df)
    # df = pd.concat(df, axis=0, ignore_index=True)

    sns.set_style('darkgrid')
    plt.figure(figsize=(8, 6))
    plt.rc('axes', labelsize=20)
    plt.rc('xtick', labelsize=18)
    plt.rc('ytick', labelsize=18)
    plt.rc('legend', fontsize=16)
    plt.rc('font', size=20)
    c = sns.color_palette()
    ax = sns.lineplot(data=df, x='# edges', y='jsd', hue='method',
                      style='method', markers=False, linewidth=2.5, markersize=13, ci=None,
                      palette=c[:3], dashes=[''] * 3)

    labels = ax.get_legend_handles_labels()

    ax.set_ylabel('JSD')
    ax.set_xlabel('Number of unlearned edges')
    # ax.set_xticklabels([f'{int(x / dataset_num_edges[args.data] * 100)}%' for x in df['# edges'].unique()])
    # ax.set_xticklabels(list(range(200, 2200, 200)))
    ax.set_xticks([100, 200, 300, 400, 500])
    # ax.set_ylim(0, 0.1)
    plt.subplots_adjust(left=0.17, bottom=0.14)
    plt.legend([], [], frameon=False)
    plt.savefig(os.path.join('./plot/', f'rq2_jsd_{args.data}_{args.model}.pdf'), bbox_inches='tight')
    plt.show()

    fig_legend, ax_legend = plt.subplots(figsize=(10, 1))
    ax_legend.axis(False)
    ax_legend.legend(
        *labels,
        # [labels[0][0], labels[0][4], labels[0][1], labels[0][5], labels[0][2], labels[0][6], labels[0][3], labels[0][7]],
        # [labels[1][0], labels[1][4], labels[1][1], labels[1][5], labels[1][2], labels[1][6], labels[1][3], labels[1][7]],
        loc='center', ncol=3)
    fig_legend.savefig(os.path.join('plot', 'rq2_jsd_legend.pdf'), bbox_inches='tight')


def _rq1_efficacy_jsd(args):
    df_list = []
    for m in ['gcn', 'sage', 'gin']:
        # df = pd.read_csv(os.path.join(f'./result/appr_posterior_{args.data}_{m}.csv'))
        df = pd.read_csv(os.path.join('./result', f'rq1_efficacy_jsd_{args.data}_{m}.csv'))
        df = df[df['# edges'].isin([100, 200, 400, 800, 1000])]
        df['# edges'] = df['# edges'].values.astype(str)
        df['target'] = np.array([m] * df.shape[0])
        df_list.append(df)
    df = pd.concat(df_list, ignore_index=True)
    print(df)

    sns.set_style('darkgrid')
    plt.figure(figsize=(8, 6))
    plt.rc('axes', labelsize=20)
    plt.rc('xtick', labelsize=20)
    plt.rc('ytick', labelsize=20)
    plt.rc('legend', fontsize=16)
    plt.rc('font', size=20)
    c = sns.color_palette()
    ax = sns.lineplot(data=df, x='# edges', y='JSD', hue='target',
                      style='target', markers=True, linewidth=2.5, markersize=13, ci=None,
                      palette=[c[0], c[2], c[3]])

    labels = ax.get_legend_handles_labels()

    # ax.set_ylabel
    ax.set_xlabel('Number of unlearned edges')
    # ax.set_xticks([1, 2, 3, 4, 5])
    # ax.set_xticklabels(['100', '200', '400', '800', '1000'])
    # ax.set_xticklabels([f'{int(x / dataset_num_edges[args.data] * 100)}%' for x in df['# edges'].unique()])
    # ax.set_ylim(0.01, 0.08)
    plt.legend([], [], frameon=False)
    plt.savefig(os.path.join('./plot/', f'posterior_{args.data}.pdf'), bbox_inches='tight')
    plt.show()

    fig_legend, ax_legend = plt.subplots(figsize=(10, 1))
    ax_legend.axis(False)
    ax_legend.legend(labels[0], ['GCN', 'GraphSAGE', 'GIN'], loc='center', ncol=4)
    fig_legend.savefig(os.path.join('plot', 'posterior_legend.pdf'), bbox_inches='tight')


def rq1_fidelity(args):
    # if args.feature:
    #     unlearn_df = pd.read_csv(os.path.join('./result', f'rq1_fidelity_{args.data}_{args.model}.csv'))
    #     baseline_df = pd.read_csv(os.path.join('./result', f'rq1_fidelity_baseline_{args.data}_{args.model}.csv'))
    # else:
    unlearn_df = pd.read_csv(os.path.join('./result', f'rq1_fidelity_{args.data}_{args.model}_no-feature.csv'))
    baseline_df = pd.read_csv(os.path.join(
        './result', f'rq1_fidelity_baseline_{args.data}_{args.model}_no-feature.csv'))

    # df = pd.DataFrame({
    #     '# edges': unlearn_df['# edges'],
    #     'Setting': unlearn_df['setting'],
    #     'Model accuracy': unlearn_df['accuracy'],
    # })
    df = pd.DataFrame({
        '# edges': baseline_df['# edges'].append(unlearn_df['# edges'], ignore_index=True),
        'Setting': baseline_df['partition'].str.upper().append(unlearn_df['setting'], ignore_index=True),
        'Model accuracy': baseline_df['accuracy'].append(unlearn_df['accuracy'], ignore_index=True),
    })
    df['# edges'] = df['# edges'].values.astype(str)

    print(df)

    sns.set_style('darkgrid')
    plt.figure(figsize=(10, 6))
    plt.rc('axes', labelsize=20)
    plt.rc('xtick', labelsize=18)
    plt.rc('ytick', labelsize=18)
    plt.rc('legend', fontsize=16)
    plt.rc('font', size=20)
    c = sns.color_palette()
    ax = sns.lineplot(data=df, x='# edges', y='Model accuracy', hue='Setting',
                      style='Setting', markers=True, linewidth=2.5, markersize=13, ci=None,
                      palette=[c[0], 'black', c[2], c[3]])

    labels = ax.get_legend_handles_labels()

    ax.set_xlabel('Number of unlearned edges')
    ax.set_ylabel('Model accuracy')
    # ax.set_xticks([0, 100, 200, 400, 800, 1000])
    # ax.set_xticklabels([f'{int(x / dataset_num_edges[args.data] * 100)}%' for x in df['# edges'].unique()])
    # ax.set_xticks(list(range(0, 2200, 200)))
    plt.legend([], [], frameon=False)
    figure_filename = f'rq1_fidelity_{args.data}_{args.model}.pdf'
    plt.savefig(os.path.join('./plot/', figure_filename), bbox_inches='tight')
    plt.show()

    fig_legend, ax_legend = plt.subplots(figsize=(10, 1))
    ax_legend.axis(False)
    ax_legend.legend(labels[0], ['BLPA', 'BEKM', 'Retrain', 'EraEdge'], loc='center', ncol=5)
    fig_legend.savefig(os.path.join('./plot', f'rq1_fidelity_legend.pdf'), bbox_inches='tight')


def RQ1_utility_comparition(args):
    if args.hidden:
        unlearn_df = pd.read_csv(os.path.join(
            './result', f'rq1_unlearn_{args.data}_{args.model}_l{len(args.hidden)}_{"_".join(map(str, args.hidden))}.csv'), index_col=0)
        baseline_df = pd.read_csv(os.path.join(
            './result', f'rq1_baseline_{args.data}_{args.model}_h{len(args.hidden)}_{"_".join(map(str, args.hidden))}.csv'), index_col=0)
    else:
        unlearn_df = pd.read_csv(os.path.join('./result', f'rq1_unlearn_{args.data}_{args.model}.csv'), index_col=0)
        baseline_df = pd.read_csv(os.path.join('./result', f'rq1_retrain_{args.data}_{args.model}.csv'), index_col=0)

    df = pd.concat((unlearn_df, baseline_df), axis=1)
    retrain_acc = df['retrain-acc']
    retrain_time = df['retrain-time']
    blpa_acc = df['blpa-acc']
    blpa_time = df['blpa-time'] / df['blpa-#-shards']
    bekm_acc = df['bekm-acc']
    bekm_time = df['bekm-time'] / df['bekm-#-shards']
    unlearn_acc = df['unlearn-acc']
    unlearn_time = df['unlearn-time']

    sns.set_style('whitegrid')
    plt.rc('axes', labelsize=22)
    plt.rc('xtick', labelsize=24)
    plt.rc('ytick', labelsize=24)
    # plt.rc('legend', fontsize=16)
    plt.rc('font', size=20)

    plt.figure(figsize=(10, 10))
    ax = plt.axes(projection='3d')
    ax.xaxis.labelpad = 12
    ax.yaxis.labelpad = 12
    ax.zaxis.labelpad = 12
    ax.plot(df.index, retrain_acc, retrain_time, '*-', label='Retrain', linewidth=2.5, markersize=13)
    ax.plot(df.index, blpa_acc, blpa_time, '^--', label='BLPA', linewidth=2.5, markersize=13)
    ax.plot(df.index, bekm_acc, bekm_time, 'v-.', label='BEKM', linewidth=2.5, markersize=13)
    ax.plot(df.index, unlearn_acc, unlearn_time, 'o:', label='ERAEDGE', linewidth=2.5, markersize=13)

    ax.set_xticks([200, 600, 1000, 1400, 1800])
    ax.set_xticklabels([f'{int(x/dataset_num_edges[args.data] * 100)}%' for x in [200, 600, 1000, 1400, 1800]])
    ax.set_xlabel('% of unlearned edges', linespacing=3.4)
    ax.set_ylabel('Model accuracy', linespacing=3.2)
    ax.set_zlabel('Unlearning time (seconds)', linespacing=3.2)
    # ax.set(xticks=df.index, xlabel='Number of edges to forget', ylabel='Accuracy')
    # ax.legend()
    # plt.subplots_adjust(top=-0.1)
    # plt.tight_layout()
    ax.view_init(elev=20, azim=-40)

    if args.hidden:
        figure_filename = f'rq1_{args.data}_{args.model}_h{len(args.hidden)}_{"_".join(map(str, args.hidden))}.pdf'
    else:
        figure_filename = f'rq1_{args.data}_{args.model}.pdf'
    plt.savefig(os.path.join('./plot', figure_filename), dpi=400, bbox_inches='tight')
    plt.show()


def RQ1_legend(args):
    import numpy as np
    x = np.linspace(1, 100, 1000)
    y = np.log(x)
    y1 = np.sin(x)
    y2 = np.cos(x)
    y3 = x
    sns.set_style('whitegrid')
    plt.rc('legend', fontsize=20)
    fig = plt.figure("Line plot")
    legendFig = plt.figure("Legend plot", figsize=(10, 1))
    ax = fig.add_subplot(111)
    line1, = ax.plot(x, y, '*-', lw=2.5, markersize=13)
    line2, = ax.plot(x, y1, '^--', lw=2.5, markersize=13)
    line3, = ax.plot(x, y2, 'v-.', lw=2.5, markersize=13)
    line4, = ax.plot(x, y3, 'o:', lw=2.5, markersize=13)
    legendFig.legend([line1, line2, line3, line4], ["Retrain", "BLPA", "BEKM", "ERAEDGE"], loc='center', ncol=4)
    legendFig.savefig(os.path.join('./plot', 'rq1_legend.pdf'))


def node_unlearn_legend(args):
    import numpy as np
    x = np.linspace(1, 100, 1000)
    y = np.log(x)
    y1 = np.sin(x)
    sns.set_style('whitegrid')
    plt.rc('legend', fontsize=20)
    fig = plt.figure("Line plot")
    legendFig = plt.figure("Legend plot", figsize=(10, 1))
    ax = fig.add_subplot(111)
    line1, = ax.plot(x, y, '*-', lw=2.5, markersize=13)
    line2, = ax.plot(x, y1, 'o:', lw=2.5, markersize=13)
    legendFig.legend([line1, line2], ["Retrain", "ours"], loc='center', ncol=2)
    legendFig.savefig(os.path.join('./plot', 'node_unlearn_legend.pdf'))


def RQ0_effectiveness(args):
    df = pd.read_csv(os.path.join('./result', f'rq0_{args.data}.csv'))

    sns.set_theme(style='darkgrid')
    plt.figure(figsize=(10, 6))
    plt.rc('axes', labelsize=20)
    plt.rc('xtick', labelsize=16)
    plt.rc('ytick', labelsize=16)
    plt.rc('legend', fontsize=16)
    plt.rc('font', size=20)

    sns.lineplot(x='# edges', y='eculiden-distance', data=df, hue='target model',
                 linewidth=2.5, style='target model', markers=True,
                 #  palette=sns.color_palette('Paired', 8),
                 markersize=13)
    # c = sns.color_palette('Paired', 8)
    # sns.lineplot(x='# edges', y='random-R', data=df, palette=c[0], label='Random-R', linewidth=2.5, markers=True)
    # sns.lineplot(x='# edges', y='random-U', data=df, palette=c[1], label='Random-U', linewidth=2.5)
    # sns.lineplot(x='# edges', y='max-degree-R', data=df, palette=c[2], label='MaxDegree-R', linewidth=2.5)
    # sns.lineplot(x='# edges', y='max-degree-U', data=df, palette=c[3], label='MaxDegree-U', linewidth=2.5)
    # sns.lineplot(x='# edges', y='min-degree-R', data=df, palette=c[4], label='MinDegree-R', linewidth=2.5)
    # sns.lineplot(x='# edges', y='min-degree-U', data=df, palette=c[5], label='MinDegree-U', linewidth=2.5)
    # sns.lineplot(x='# edges', y='saliency-R', data=df, palette=c[6], label='Saliency-R', linewidth=2.5)
    # sns.lineplot(x='# edges', y='saliency-U', data=df, palette=c[7], label='Saliency-U', linewidth=2.5)

    # plt.xticks(df['# edges'], [f'{int(x/dataset_num_edges[args.data] * 100)}%' for x in df['# edges']])
    # plt.xticklabels([f'{int(x/dataset_num_edges[args.data] * 100)}%' for x in df['# edges']])
    # plt.xlabel('Number of edges')
    # plt.legend(ncol=2)
    # plt.subplots_adjust(bottom=0.15)
    plt.savefig(os.path.join('./plot', f'rq0_{args.data}.pdf'), dpi=400, bbox_inches='tight')
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', dest='data', type=str, required=True,
                        help='The dataset you want to show.')
    parser.add_argument('-m', dest='model', type=str, default=None,
                        help='The target model you want to visualize.')
    parser.add_argument('-hidden', type=int, nargs='+', default=[])
    parser.add_argument('-rq', type=str, default=None)
    parser.add_argument('-feature', action='store_true')
    # parser.add_argument('-rq2', action='store_true')
    # parser.add_argument('-rq2-legend', action='store_true')
    # parser.add_argument('-rq3', action='store_true')
    # parser.add_argument('-rq4-1', action='store_true')
    # parser.add_argument('-rq4-2', action='store_true')
    # parser.add_argument('-node-unlearn', action='store_true')
    args = parser.parse_args()

    print('Argument:', vars(args))

    if args.rq == 'rq2_fidelity':
        _rq2_fidelity(args)

    if args.rq == 'rq2_efficiency':
        _rq2_efficiency(args)

    if args.rq == 'rq2_jsd':
        _rq2_jsd(args)

    if args.rq == 'adv1':
        RQ4_adversarial_edges_unlearn(args)

    if args.rq == 'adv2':
        RQ4_adversarial_vs_benign(args)

    if args.rq == 'loss':
        _approximation_evaluate(args)

    if args.rq == 'rq1_fidelity':
        rq1_fidelity(args)

    if args.rq == 'rq1_efficiency':
        rq1_efficiency(args)

    if args.rq == 'rq1_efficacy':
        _rq1_efficacy_jsd(args)

    if args.rq == 'rq3_jsd':
        _rq3_jsd(args)

    if args.rq == 'rq4_fidelity':
        _rq4_fidelity(args)

    # if args.rq1 == 0:
    #     RQ1_legend(args)
    # elif args.rq1 == 1:
    #     # RQ1_utility_comparition(args)
    #     RQ1_target_model_utility(args)
    # elif args.rq1 == 2:
    #     RQ1_running_time(args)

    # if args.rq4:
    #     RQ2_gnn(args)

    # if args.rq2_legend:
    #     RQ2_legend(args)

    # if args.rq2:
    #     RQ3_edges_sampling(args)

    # if args.rq4_1:
    #     RQ4_adversarial_edges_unlearn(args)

    # if args.rq4_2:
    #     RQ4_adversarial_vs_benign(args)

    # if args.node_unlearn:
    #     node_unlearn(args)
