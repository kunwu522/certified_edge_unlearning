import os
import time
import torch
import numpy as np
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, accuracy_score, roc_curve, precision_recall_curve, average_precision_score, auc


def influence_matrix(nodes, edge_index, model, v, delta):
    model.eval()
    with torch.no_grad():
        P = model(nodes, edge_index)
        P_prime = model(nodes, edge_index, v, delta)
    I = (1 / delta) * (P_prime - P)
    return I.cpu().numpy()


def linkteller_attack(mode, nodes, features, edge_index, model, exist_edges, nonexist_edges, influence=1e-5, device=None):
    attacker = Attacker(mode, nodes, features, edge_index, model, influence, exist_edges, nonexist_edges, device)
    return attacker.link_prediction_attack()


def _arg_nearest_mean(a):
    # calculate the difference array
    difference_array = np.absolute(a - np.mean(a))

    # find the index of minimum element from the array
    index = difference_array.argmin()
    # print("Nearest element to the given values is : ", arr[index])
    # print("Index of nearest value is : ", index)
    return index


class Attacker:
    def __init__(self, mode, nodes, features, edge_index, model, influence, exist_edges, nonexist_edges, device):
        self.mode = mode
        self.model = model
        self.device = device
        self.influence = influence
        self.exist_edges = exist_edges
        self.nonexist_edges = nonexist_edges

        # self.nodes = torch.tensor(nodes, device=device)
        self.nodes = nodes
        self.features = torch.from_numpy(features).to(device)
        self.edge_index = edge_index
        self.verbose = False

    # \partial_f(x_u) / \partial_x_v
    def get_gradient(self, u, v):
        h = 0.0001
        ret = torch.zeros(self.features.shape[1])
        for i in range(self.features.shape[1]):
            pert = torch.zeros_like(self.features, device=device)
            pert[v][i] = h
            with torch.no_grad():
                grad = (self.model(self.nodes, self.edge_index, v, pert).detach() -
                        self.model(self.nodes, self.edge_index, v, -pert).detach()) / (2 * h)
                ret[i] = grad[u].sum()

        return ret

    # \partial_f(x_u) / \partial_epsilon_v
    def get_gradient_eps(self, u, v):
        pert_1 = torch.zeros_like(self.features)

        pert_1[v] = self.features[v] * self.influence

        grad = (self.model(self.nodes, self.edge_index, v, pert_1).detach() -
                self.model(self.nodes, self.edge_index).detach()) / self.influence

        return grad[u]

    # def calculate_auc(self, v1, v0):
    #     v1 = sorted(v1)
    #     v0 = sorted(v0)
    #     vall = sorted(v1 + v0)

    #     TP = 500
    #     # FP = 500
    #     FN = 0
    #     TN = 500
    #     T = N = 500  # fixed

    #     p0 = p1 = 0

    #     TPR = TP / T
    #     # FPR = FP / F
    #     TNR = TN / N

    #     result = [(TNR, TPR)]
    #     auc = 0
    #     for elem in vall:
    #         if p1 < 500 and abs(elem - v1[p1]) < 1e-6:
    #             p1 += 1
    #             TP -= 1
    #             TPR = TP / T
    #         else:
    #             p0 += 1
    #             FP -= 1
    #             FPR = FP / F
    #             auc += TPR * 1 / F

    #         result.append((FPR, TPR))

    #     return result, auc

    def link_prediction_attack(self):
        norm_exist = []
        norm_nonexist = []

        with torch.no_grad():
            t = time.time()
            exist_iterator = tqdm(self.exist_edges) if self.verbose else self.exist_edges
            for u, v in exist_iterator:
                grad = self.get_gradient_eps(u, v)  # if self.args.approx else self.get_gradient(u, v)
                norm_exist.append(grad.norm().item())

            if self.verbose:
                print(f'time for predicting existing edges: {time.time() - t}')

            t = time.time()
            nonexist_iterator = tqdm(self.nonexist_edges) if self.verbose else self.nonexist_edges
            for u, v in nonexist_iterator:
                grad = self.get_gradient_eps(u, v)  # if self.args.approx else self.get_gradient(u, v)
                norm_nonexist.append(grad.norm().item())

            if self.verbose:
                print(f'time for predicting non-existing edges: {time.time() - t}')

        y = [1] * len(norm_exist) + [0] * len(norm_nonexist)
        pred = norm_exist + norm_nonexist

        # sorted_pred = np.sort(pred)
        # idx_mean = _arg_nearest_mean(sorted_pred)

        y_pred = np.ones_like(pred)
        y_pred[np.argsort(pred)[:int(len(pred)/2)]] = 0
        tn, fp, fn, tp = confusion_matrix(y, y_pred).ravel()

        acc = (tp + tn) / (tp + fp + tn + fn)
        tpr = tp / (tp + fn)
        tnr = tn / (tn + fp)

        return acc, tpr, 1 - tnr
