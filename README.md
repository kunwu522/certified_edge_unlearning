# Certified Edge Unlearning for Graph Neural Networks
This is an implementation of paper [Certified Edge Unlearning for Graph Neural Networks](). We provide the code of graph edge unlearning. In addition, we present the code for reproducing the experiments.

## Dependencies
* tqdm/numpy/pandas/sklearn/matplotlib/seaborn
* Pytorch >= 1.09
* pyg (torch_geometric)
* [DeepRobust](https://deeprobust.readthedocs.io/en/latest/)
* [StellarGraph](https://github.com/stellargraph/stellargraph)

## Result Reproduction
### Tightness of Bounds
```
python experiment.py -rq bound
```
### Accuracy of CEU
```
python experiment.py -rq unlearn
```
### Efficency of CEU
```
python experiment.py -rq efficiency
```
### Efficacy of CEU
```
python experiment.py -rq efficacy
```

### Effect of $\epsilon$
```
python experiment.py --rq epsilon
```

### CGU Comparison
```
python experiment.py --rq cgu_compare
```


### Common Parameters
* -g, the ID of a GPU you want to use. Default: -1 (using CPU)
* -edges, a list, indicates the numbers of edges you want to unlearn. Default: \[100, 200, 400, 800, 1000\].
* -targets, a list, indicates what target models you want to evaluate. Default:\['gcn', 'gat', 'sage', 'gin'\].
* -datasets, a list, indicate what datasets you want to use. Default:\['cora', 'citeseer', 'cs', 'physics\].

