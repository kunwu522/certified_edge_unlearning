import argparse
from email.policy import default


def argument_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-seed', type=int, default=522)

    # Task
    parser.add_argument('-data-info', action='store_true')
    parser.add_argument('-model-info', action='store_true')
    parser.add_argument('-train', action='store_true', help='Indicator of training.')
    parser.add_argument('-retrain', action='store_true', help='Indicator of retraining.')
    parser.add_argument('-unlearn', action='store_true', help='Indicator of unlearning.')
    parser.add_argument('-influence', action='store_true',
                        help='Indicator of running a experiment on "influence vs loss difference".')
    parser.add_argument('-epsilon', action='store_true',
                        help='Indicator of calculating the epsilon value for each edge.')

    parser.add_argument('-retrain-node', action='store_true', help='Indicator of retraining.')
    parser.add_argument('-infl-node', dest='influence_node', action='store_true',
                        help='Indicator of running a experiment on "influence vs loss difference".')

    parser.add_argument('-save', action='store_true', help='save the result to a file.')

    # Evaluation Task
    parser.add_argument('-l', dest='loss_diff', action='store_true',
                        help='Indicator of running analysis on influence against loss difference')
    parser.add_argument('-i', dest='inference_comparison', action='store_true',
                        help='Indicator of evaluating the unlearning model by inference comparison.')
    # parser.add_argument('-mia', dest='mia_attack', action='store_true',
    #                     help='Indicator of evaluting the unlearning model via MIA attack (accuracy).')
    parser.add_argument('-l2-dis', dest='l2_distance', action='store_true',
                        help='Indicator of evaluating the unlearning by l2 distance.')
    parser.add_argument('-test-mia', action='store_true')
    parser.add_argument('-performance', dest='performance', action='store_true')
    parser.add_argument('-performance-type', dest='performance_type', type=str, default='original')
    parser.add_argument('-condition-number', action='store_true')
    parser.add_argument('-adv', action='store_true')

    # Common arugment
    parser.add_argument('-g', dest='gpu', type=int, default=-1)
    parser.add_argument('-d', dest='data', type=str, default='cora')
    parser.add_argument('-m', dest='model', type=str, default='gcn')
    parser.add_argument('-verbose', action='store_true')

    # For training
    parser.add_argument('-hidden', type=int, nargs='+', default=[])
    parser.add_argument('-epochs', type=int, default=1000)
    parser.add_argument('-batch', type=int, default=512)
    parser.add_argument('-test-batch', type=int, default=1024)
    parser.add_argument('-lr', type=float, default=0.001)
    parser.add_argument('-l2', type=float, default=1E-5)
    parser.add_argument('-emb-dim', type=int, default=32)
    parser.add_argument('-feature', dest='feature', action='store_true')
    parser.add_argument('-no-feature-update', dest='feature_update', action='store_false')
    parser.add_argument('-p', dest='patience', type=int, default=20)
    parser.add_argument('-no-early-stop', dest='early_stop', action='store_false')

    # For unlearning
    parser.add_argument('-approx', type=str, default='cg')
    parser.add_argument('-method', type=str, default='random')
    parser.add_argument('-max-degree', action='store_true')
    parser.add_argument('-edges', type=int, nargs='+', default=[100, 200, 400, 800, 1000],
                        help='in terms of precentage, how many edges to sample.')
    parser.add_argument('-batch-unlearn', action='store_true')
    parser.add_argument('-unlearn-batch-size', type=int, default=None)

    # unlearning parameters
    parser.add_argument('-damping', type=float, default=0.)
    parser.add_argument('-eps', type=float, default=1E-5)

    # parameters for Lissa approximation
    parser.add_argument('-depth', type=int, default=300)
    parser.add_argument('-r', type=int, default=10)
    parser.add_argument('-scale', type=int, default=1)

    return parser
