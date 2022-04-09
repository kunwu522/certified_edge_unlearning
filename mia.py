import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_score, recall_score, roc_auc_score, accuracy_score
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


def _build_features(edges, features, posterior, verbose=False):
    # print('!!!!!!!!!', verbose)
    mia_features = []
    for edge in edges:
        v1, v2 = edge
        post1, post2 = posterior[v1], posterior[v2]
        feat1, feat2 = features[v1], features[v2]
        try:
            similarities = [metric(post1, post2) for metric in similarity_list]
        except RuntimeWarning as err:
            print('Error:', err)
            print(post1, post2)
            print(similarities)
            exit(0)

        entr1, entr2 = _entropy(post1 - post1.min()), _entropy(post2 - post2.min())

        try:
            feature_similarities = [metric(feat1, feat2) for metric in similarity_list]
        except RuntimeWarning as err:
            print('Error:', err)
            print(post1, post2)
            print(similarities)
            exit(0)

        _features = np.concatenate([post1, post2, similarities, np.array([entr1, entr2])])
        if verbose:
            print('nodes:', v1, v2)
            print('post:', post1, post2)
            print('feat:', feat1, feat2)
            print('output:', _features)

        mia_features.append(_features.reshape(1, -1))
    return np.concatenate(mia_features, axis=0)


def MIA_accuracy(edges, labels, edges_test, labels_test, features, posterior, verbose=False):
    x_train = _build_features(edges, features, posterior)
    y_train = labels
    x_test = _build_features(edges_test, features, posterior)
    y_test = labels_test

    # print('The number of inf:', np.sum(np.isinf(x_train)))
    x_train[np.isinf(x_train)] = 0
    x_test[np.isinf(x_test)] = 0

    x_train = x_train.astype('float')
    x_test = x_test.astype('float')

    ss = StandardScaler()
    x_train = ss.fit_transform(x_train)
    x_test = ss.fit_transform(x_test)

    x_train = np.nan_to_num(x_train)
    x_test = np.nan_to_num(x_test)

    mlp = MLPClassifier(solver='adam', batch_size=batch_size, alpha=1e-5, hidden_layer_sizes=(32, 32), random_state=1,
                        max_iter=50, early_stopping=True, )
    mlp.fit(x_train, y_train)

    y_pred = mlp.predict(x_test)
    test_acc = accuracy_score(y_test, y_pred)

    return test_acc
