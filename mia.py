from collections import defaultdict
import random
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_score, recall_score, roc_auc_score, accuracy_score, confusion_matrix
from scipy.spatial.distance import cosine, euclidean, correlation, chebyshev,\
    braycurtis, canberra, cityblock, sqeuclidean


similarity_list = [cosine, euclidean, correlation, chebyshev,
                   braycurtis, canberra, cityblock, sqeuclidean]
num_classes = 2
batch_size = 64
epochs = 50


def _average(a, b):
    return (a + b) / 2


def _hadamard(a, b):
    return a * b


def _weighted_l1(a, b):
    return abs(a - b)


def _weighted_l2(a, b):
    return abs((a - b) * (a - b))


def _concate_all(a, b):
    return np.concatenate(
        (_average(a, b), _hadamard(a, b), _weighted_l1(a, b), _weighted_l2(a, b)))


def _entropy(P):
    """ Epsilon is used here to avoid conditional code for
    checking that neither P nor Q is equal to 0. """
    epsilon = 0.00001
    # You may want to instead make copies to avoid changing the np arrays.
    P = P + epsilon
    entropy_value = -np.sum(P * np.log(P))
    return entropy_value


def build_features(edges, posterior, node_features=None, verbose=False):
    mia_features = []
    for edge in edges:
        v1, v2 = edge
        post1, post2 = posterior[v1], posterior[v2]
        similarities = [metric(post1, post2) for metric in similarity_list]
        entr1, entr2 = _entropy(post1), _entropy(post2)

        if node_features is not None:
            feat1, feat2 = node_features[v1], node_features[v2]
            feature_similarities = [metric(feat1, feat2) for metric in similarity_list]
            _features = np.concatenate([post1, post2, similarities, np.array(
                [entr1, entr2]), feature_similarities])
        else:
            _features = np.concatenate([post1, post2, similarities, np.array(
                [entr1, entr2])])
        # if verbose:
        #     print('nodes:', v1, v2)
        #     print('post:', post1, post2)
        #     print('feat:', feat1, feat2)
        #     print('output:', _features)

        mia_features.append(_features.reshape(1, -1))
    return np.concatenate(mia_features, axis=0)


def sample_non_member(partial_graph, num_non_members):
    print(f'Sample {num_non_members} non-member edges...')

    iteration_count = 0
    non_member_edges = []
    while len(non_member_edges) < num_non_members:
        iteration_count += 1
        if iteration_count % 10000 == 0:
            print(f'Iteration: {iteration_count}', len(non_member_edges))

        v1 = random.choice(partial_graph['nodes'])
        v2 = random.choice(partial_graph['nodes'])
        if v1 == v2:
            continue
        if (v1, v2) in partial_graph['edges'] or (v2, v1) in partial_graph['edges']:
            continue
        non_member_edges.append((v1, v2))
    return non_member_edges


def sample_member(edges, num_edges):
    return random.sample(edges, num_edges)


def sample_partial_graph(edges_prime, edges_to_forget, sample_size=0.4):
    # need to remove undirected edges
    _edges = []
    for v1, v2 in edges_prime:
        if (v1, v2) in _edges or (v2, v1) in _edges:
            continue
        _edges.append((v1, v2))

    partial_edges = sample_member(_edges, int(sample_size * len(_edges)))

    partial_nodes = set()
    for v1, v2 in partial_edges:
        partial_nodes.add(v1)
        partial_nodes.add(v2)
    for v1, v2 in edges_to_forget:
        partial_nodes.add(v1)
        partial_nodes.add(v2)

    return {'nodes:': list(partial_nodes), 'edges': partial_edges, 'saved_edges': edges_to_forget}


def construct_mia_data_original(partial_graph):
    member_edges = partial_graph['edges'][:-len(partial_graph['saved_edges'])]
    non_member_edges = sample_non_member(partial_graph, len(member_edges))

    train_edges = member_edges + non_member_edges
    train_labels = [1] * len(member_edges) + [0] * len(non_member_edges)

    random_ind = list(range(len(train_edges)))
    random.shuffle(random_ind)
    train_edges = np.array(train_edges)[random_ind]
    train_labels = np.array(train_labels)[random_ind]

    test_edges = partial_graph['saved_edges'] + sample_non_member(partial_graph)

    return train_edges, train_labels


def generate_mia_features(edges, labels, test_edges, test_labels, posterior, features=None):
    x_train = build_features(edges, posterior, features)
    y_train = labels
    x_test = build_features(test_edges, posterior, features)
    y_test = test_labels

    x_train = np.nan_to_num(x_train)
    x_test = np.nan_to_num(x_test)

    x_train = x_train.astype('float')
    x_test = x_test.astype('float')

    ss = StandardScaler()
    x_train = ss.fit_transform(x_train)
    x_test = ss.transform(x_test)

    return x_train, y_train, x_test, y_test


def MIA(x_train, y_train):

    mlp = MLPClassifier(solver='adam', batch_size=batch_size, alpha=1e-5, hidden_layer_sizes=(32, 32), random_state=1,
                        max_iter=100, early_stopping=True, )
    mlp.fit(x_train, y_train)
    return mlp

    lr = LogisticRegression().fit(x_train, y_train)
    return lr


def evaluate_mia(mlp, x_test, y_test):
    y_pred = mlp.predict(x_test)
    test_acc = accuracy_score(y_test, y_pred)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    tpr = tp / (tp+fn)
    # fpr = fp / (fp+tn)
    tnr = tn / (tn + fp)

    return test_acc, tpr.item(), tnr.item()


def MIA_attack(edges, labels, edges_test, labels_test, posterior, features=None, verbose=False):
    x_train, y_train, x_test, y_test = generate_mia_features(
        edges, labels, edges_test, labels_test, posterior, features)
    mlp = MIA(x_train, y_train)
    return evaluate_mia(mlp, x_test, y_test)
