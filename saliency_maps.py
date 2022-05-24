import pandas as pd
import numpy as np
import os
import pickle
from tqdm import tqdm
from stellargraph import StellarGraph
from stellargraph.mapper import FullBatchNodeGenerator
from stellargraph.layer import GCN, GAT
from tensorflow import keras
from tensorflow.keras import optimizers, losses, metrics, regularizers
from stellargraph.interpretability.saliency_maps import IntegratedGradients, IntegratedGradientsGAT, GradientSaliencyGAT

from data_loader import load_data
from argument import argument_parser


if __name__ == '__main__':
    parser = argument_parser()
    args = parser.parse_args()

    data = load_data(args)
    edge_df = pd.DataFrame({
        'source': [v1 for v1, _ in data['edges']],
        'target': [v2 for _, v2 in data['edges']]
    })

    G = StellarGraph(data['features'], edge_df)
    generator = FullBatchNodeGenerator(G, sparse=False)

    train_gen = generator.flow(data['train_set'].nodes, data['train_set'].labels)

    if args.model == 'gcn':
        gcn = GCN(
            layer_sizes=[int(16), int(data['num_classes'])],
            activations=["elu", 'softmax'],
            generator=generator,
            dropout=0.5,
            kernel_regularizer=regularizers.l2(1e-5),
        )
        x_inp, predictions = gcn.in_out_tensors()
    elif args.model == 'gat':
        gat = GAT(
            layer_sizes=[int(16), int(data['num_classes'])],
            attn_heads=1,
            generator=generator,
            bias=True,
            in_dropout=0,
            attn_dropout=0,
            activations=["elu", "softmax"],
            normalize=None,
            saliency_map_support=True,
        )
        x_inp, predictions = gat.in_out_tensors()

    # x_out = layers.Dense(units=data['num_classes'], activation='softmax')(x_out)

    model = keras.Model(inputs=x_inp, outputs=predictions)
    model.compile(optimizer=optimizers.Adam(lr=0.001),
                  loss=losses.sparse_categorical_crossentropy,
                  metrics=[metrics.sparse_categorical_accuracy])
    val_gen = generator.flow(data['valid_set'].nodes, data['valid_set'].labels)
    history = model.fit(train_gen, epochs=100, validation_data=val_gen)

    test_gen = generator.flow(data['test_set'].nodes, data['test_set'].labels)
    test_metrics = model.evaluate(test_gen)
    print("\nTest Set Metrics:")
    for name, val in zip(model.metrics_names, test_metrics):
        print("\t{}: {:0.4f}".format(name, val))

    saliency_edge_importance = np.zeros((data['num_nodes'], data['num_nodes']))
    all_gen = generator.flow(data['nodes'])
    for i in tqdm(range(len(data['test_set']))):
        node, label = data['test_set'][i]
        # all_gen = generator.flow(all_gen)
        y_pred = model.predict(all_gen)
        y_pred = model.predict(all_gen)[0, node]
        class_of_interest = np.argmax(y_pred)
        # print(y_pred, class_of_interest)

        if args.model == 'gcn':
            int_grad_saliency = IntegratedGradients(model, train_gen)
        elif args.model == 'gat':
            int_grad_saliency = IntegratedGradientsGAT(model, train_gen, generator.node_list)
            saliency = GradientSaliencyGAT(model, train_gen)
        try:
            integrate_link_importance = int_grad_saliency.get_integrated_link_masks(node, class_of_interest, steps=50)
        except AttributeError as err:
            # print('Error:', err)
            continue
        saliency_edge_importance += integrate_link_importance
        # print("integrate_link_importance.shape = {}".format(integrate_link_importance.shape))
    ind_row, ind_col = np.unravel_index(np.argsort(saliency_edge_importance, axis=None), saliency_edge_importance.shape)
    sorted_edges = np.concatenate((ind_row[::-1], ind_col[::-1])).reshape(2, -1).T.tolist()

    print('sorted_edges:', sorted_edges[:20])

    valid_edges = []
    for edge in tqdm(sorted_edges):
        if edge in data['edges']:
            valid_edges.append(edge)

    sorted_saliency_edges_path = os.path.join('./data', args.data, f'sorted_saliency_edges_{args.model}.list')
    with open(sorted_saliency_edges_path, 'wb') as fp:
        pickle.dump(sorted_edges, fp)
