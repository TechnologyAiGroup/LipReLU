"""Data utils functions for pre-processing and data loading."""
import os

import networkx as nx
import numpy as np
import scipy.sparse as sp

def load_new_data(dataset_str, data_path):
    # def preprocess_features(features):
    #     """Row-normalize feature matrix and convert to tuple representation"""
    #     rowsum = np.array(features.sum(1))
    #     rowsum = (rowsum==0)*1+rowsum
    #     r_inv = np.power(rowsum, -1).flatten()
    #     r_inv[np.isinf(r_inv)] = 0.
    #     r_mat_inv = sp.diags(r_inv)
    #     features = r_mat_inv.dot(features)
    #     return features
    graph_adjacency_list_file_path = os.path.join(data_path, 'out1_graph_edges.txt')
    graph_node_features_and_labels_file_path = os.path.join(data_path, 'out1_node_feature_label.txt')

    G = nx.DiGraph()
    graph_node_features_dict = {}
    graph_labels_dict = {}

    if dataset_str == 'film':
        with open(graph_node_features_and_labels_file_path) as graph_node_features_and_labels_file:
            graph_node_features_and_labels_file.readline()
            for line in graph_node_features_and_labels_file:
                line = line.rstrip().split('\t')
                assert (len(line) == 3)
                assert (int(line[0]) not in graph_node_features_dict and int(line[0]) not in graph_labels_dict)
                feature_blank = np.zeros(932, dtype=np.uint8)
                feature_blank[np.array(line[1].split(','), dtype=np.uint16)] = 1
                graph_node_features_dict[int(line[0])] = feature_blank
                graph_labels_dict[int(line[0])] = int(line[2])

    else:
        with open(graph_node_features_and_labels_file_path) as graph_node_features_and_labels_file:
            graph_node_features_and_labels_file.readline()
            for line in graph_node_features_and_labels_file:
                line = line.rstrip().split('\t')
                assert (len(line) == 3)
                assert (int(line[0]) not in graph_node_features_dict and int(line[0]) not in graph_labels_dict)
                graph_node_features_dict[int(line[0])] = np.array(line[1].split(','), dtype=np.uint8)
                graph_labels_dict[int(line[0])] = int(line[2])

    with open(graph_adjacency_list_file_path) as graph_adjacency_list_file:
        graph_adjacency_list_file.readline()
        for line in graph_adjacency_list_file:
            line = line.rstrip().split('\t')
            assert (len(line) == 2)
            if int(line[0]) not in G:
                G.add_node(int(line[0]), features=graph_node_features_dict[int(line[0])],
                            label=graph_labels_dict[int(line[0])])
            if int(line[1]) not in G:
                G.add_node(int(line[1]), features=graph_node_features_dict[int(line[1])],
                            label=graph_labels_dict[int(line[1])])
            G.add_edge(int(line[0]), int(line[1]))
    adj = nx.adjacency_matrix(G, sorted(G.nodes()))
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)   # 使得邻接矩阵对称
    features = np.array([features for _, features in sorted(G.nodes(data='features'), key=lambda x: x[0])])
    labels = np.array([label for _, label in sorted(G.nodes(data='label'), key=lambda x: x[0])])
    # features = preprocess_features(features)
    return sp.csr_matrix(adj), sp.csr_matrix(features), labels

# load_new_data('texas', 'data/texas')   # 示例使用

# """Data utils functions for pre-processing and data loading."""
# import os
#
# import networkx as nx
# import numpy as np
# import scipy.sparse as sp
#
# def load_new_data(dataset_str, data_path):
#     def preprocess_features(features):
#         """Row-normalize feature matrix and convert to tuple representation"""
#         rowsum = np.array(features.sum(1))
#         rowsum = (rowsum==0)*1+rowsum
#         r_inv = np.power(rowsum, -1).flatten()
#         r_inv[np.isinf(r_inv)] = 0.
#         r_mat_inv = sp.diags(r_inv)
#         features = r_mat_inv.dot(features)
#         return features
#     graph_adjacency_list_file_path = os.path.join(data_path, 'out1_graph_edges.txt')
#     graph_node_features_and_labels_file_path = os.path.join(data_path, 'out1_node_feature_label.txt')
#
#     G = nx.DiGraph()
#     graph_node_features_dict = {}
#     graph_labels_dict = {}
#
#     if dataset_str == 'film':
#         with open(graph_node_features_and_labels_file_path) as graph_node_features_and_labels_file:
#             graph_node_features_and_labels_file.readline()
#             for line in graph_node_features_and_labels_file:
#                 line = line.rstrip().split('\t')
#                 assert (len(line) == 3)
#                 assert (int(line[0]) not in graph_node_features_dict and int(line[0]) not in graph_labels_dict)
#                 feature_blank = np.zeros(932, dtype=np.uint8)
#                 feature_blank[np.array(line[1].split(','), dtype=np.uint16)] = 1
#                 graph_node_features_dict[int(line[0])] = feature_blank
#                 graph_labels_dict[int(line[0])] = int(line[2])
#
#     else:
#         with open(graph_node_features_and_labels_file_path) as graph_node_features_and_labels_file:
#             graph_node_features_and_labels_file.readline()
#             for line in graph_node_features_and_labels_file:
#                 line = line.rstrip().split('\t')
#                 assert (len(line) == 3)
#                 assert (int(line[0]) not in graph_node_features_dict and int(line[0]) not in graph_labels_dict)
#                 graph_node_features_dict[int(line[0])] = np.array(line[1].split(','), dtype=np.uint8)
#                 graph_labels_dict[int(line[0])] = int(line[2])
#
#     with open(graph_adjacency_list_file_path) as graph_adjacency_list_file:
#         graph_adjacency_list_file.readline()
#         for line in graph_adjacency_list_file:
#             line = line.rstrip().split('\t')
#             assert (len(line) == 2)
#             if int(line[0]) not in G:
#                 G.add_node(int(line[0]), features=graph_node_features_dict[int(line[0])],
#                             label=graph_labels_dict[int(line[0])])
#             if int(line[1]) not in G:
#                 G.add_node(int(line[1]), features=graph_node_features_dict[int(line[1])],
#                             label=graph_labels_dict[int(line[1])])
#             G.add_edge(int(line[0]), int(line[1]))
#     adj = nx.adjacency_matrix(G, sorted(G.nodes()))
#     features = np.array([features for _, features in sorted(G.nodes(data='features'), key=lambda x: x[0])])
#     labels = np.array([label for _, label in sorted(G.nodes(data='label'), key=lambda x: x[0])])
#     features = preprocess_features(features)
#     return sp.csr_matrix(adj), features, labels