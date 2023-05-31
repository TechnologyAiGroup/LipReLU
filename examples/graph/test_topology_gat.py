import torch
import numpy as np
import torch.nn.functional as F
from deeprobust.graph.defense.gcn_lip import GCN
from deeprobust.graph.data import Dataset, Dpr2Pyg
from deeprobust.graph.global_attack.topology_attack_gat import MinMax
from deeprobust.graph.utils import *
from deeprobust.graph.data import Dataset
from deeprobust.graph.utils import preprocess
from deeprobust.graph.defense.gat_lip_topo_pyg import GAT
import argparse
import random


parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=15, help='Random seed.')
# parser.add_argument('--seed', type=int, default=25, help='Random seed.')
parser.add_argument('--epochs', type=int, default=200,
                    help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.01,
                    help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=5e-4,
                    help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden', type=int, default=16,
                    help='Number of hidden units.')
parser.add_argument('--dropout', type=float, default=0.5,
                    help='Dropout rate (1 - keep probability).')

parser.add_argument('--dataset', type=str, default='citeseer', choices=['cora', 'cora_ml', 'citeseer', 'polblogs', 'pubmed'], help='dataset')
parser.add_argument('--ptb_rate', type=float, default=0.25,  help='pertubation rate')    # 0.23替换0.25
# parser.add_argument('--model', type=str, default='PGD', choices=['PGD', 'min-max'], help='model variant')   #一开始默认是PGD?

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# device = 'cpu'
args = parser.parse_args()
np.random.seed(args.seed)
torch.manual_seed(args.seed)
random.seed(args.seed)
if device != 'cpu':
    torch.cuda.manual_seed(args.seed)


data = Dataset(root='/tmp/', name=args.dataset)
adj, features, labels = data.adj, data.features, data.labels
adj, features, labels = preprocess(adj, features, labels, preprocess_adj=False, device=device)
idx_train, idx_val, idx_test = data.idx_train, data.idx_val, data.idx_test

perturbations = int(args.ptb_rate * (adj.sum()//2))

# # Setup Victim Model
# victim_model = GAT(nfeat=features.shape[1],
#                nhid=8, heads=8,
#                nclass=labels.max().item() + 1,
#                dropout=0.5, device=device)
# #
# victim_model = victim_model.to(device)
# edge_index = adj.nonzero().t()
# row, col = edge_index
# edge_weight = adj[row, col]
# victim_model.fit(features, edge_index, edge_weight, labels, idx_train, idx_val, verbose=True)
# #
# # # Setup Attack Model
# #
# model = MinMax(model=victim_model, nnodes=adj.shape[0], loss_type='CE', device=device)
# model = model.to(device)
# model.attack(features, adj, labels, idx_train, perturbations)
# modified_adj = model.modified_adj
# print(modified_adj.shape)

# path_1 = "examples/graph/topology_adj/cora_topology_lip_adj_normal.pt"
# path_2 = "examples/graph/topology_adj/cora_topology_lip_modified_adj_ptb_rate_0.25.pt"
path_1 = "examples/graph/topology_modified_adj/citeseer_topology_adj_normal.pt"
path_2 = "examples/graph/topology_modified_adj/citeseer_topology_modified_adj_ptb_rate_0.25.pt"
#
# # torch.save(adj, path_1)
# # torch.save(modified_adj, path_2)
original_adj = torch.load(path_1)
modified_adj = torch.load(path_2)

# modified_adj = modified_adj.to(device)

# 将邻接矩阵修改成pyg所需的形式
edge_index = (modified_adj > 0).nonzero().t()
row, col = edge_index
edge_weight = modified_adj[row, col]

gat = GAT(nfeat=features.shape[1],
      nhid=8, heads=8,
      nclass=labels.max().item() + 1,
      dropout=0.5, device=device)

gat = gat.to(device)


# test on clean graph
print('==================')
print('=== train on clean graph ===')

gat.fit(features, edge_index, edge_weight, labels, idx_train, idx_val, lip_norm=False, lip_node=False, lip_relu=False, gamma=0.008, verbose=True) # train with earlystopping
gat.test(features, edge_index, edge_weight, labels, idx_test)
