from scipy import sparse
from ogb.nodeproppred import PygNodePropPredDataset
import torch
import numpy as np
import torch.nn.functional as F
import torch.optim as optim
from deeprobust.graph.defense.gcn_lip import GCN
from deeprobust.graph.global_attack import DICE
from deeprobust.graph.defense.gcn_preprocess_lip import GCNJaccard
from deeprobust.graph.defense.gcn_preprocess_lip import GCNSVD
from deeprobust.graph.utils import *
from deeprobust.graph.data import Dataset
import scipy
import argparse
import os
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=15, help='Random seed.')
parser.add_argument('--dataset', type=str, default='arxiv', choices=['cora', 'cora_ml', 'citeseer', 'polblogs', 'pubmed'], help='dataset')
parser.add_argument('--ptb_rate', type=float, default=0.20,  help='pertubation rate')

args = parser.parse_args()
args.cuda = torch.cuda.is_available()
print('cuda: %s' % args.cuda)
device = torch.device("cuda:7" if torch.cuda.is_available() else "cpu")

np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

def edge_index_to_coo_dense(edge_index, N):
    v = torch.ones(edge_index.size(1))
    adj_coo = torch.sparse_coo_tensor(edge_index, v, (N, N))
    return adj_coo.to_dense()



# dataset = PygNodePropPredDataset(name = 'ogbn-arxiv')
# data = dataset[0]
# adj, features, labels = data.edge_index, data.x, data.y
# split_idx = dataset.get_idx_split()
# train_idx, val_idx, test_idx = split_idx['train'], split_idx['valid'], split_idx['test']


# edge_set = {(u.item(), v.item()) if u < v else (v.item(), u.item()) for u, v in  adj.T}
# pert = np.load('examples/graph/arxiv_pertubated_adj/ogbn_arxiv_prbcd_budget_0p05_seed_0.npz')
# pert_removed_set = {(u, v) for u, v in pert['pert_removed'].T}
# pert_added_set = {(u, v) for u, v in pert['pert_added'].T}
# pert_edge_set = edge_set - pert_removed_set | pert_added_set
# pert_edge_index = torch.tensor(list(pert_edge_set)).T
#
# adj = pert_edge_index
# idx_train, idx_val, idx_test = idx_train.numpy(), idx_val.numpy(), idx_test.numpy()
# adj = edge_index_to_coo_dense(adj, features.shape[0])
# adj, features, labels = sparse.csr_matrix(adj), sparse.csr_matrix(features), labels.numpy()
# labels = labels.squeeze(-1)


# path = 'examples/graph/arxiv/ogbn_arxiv_prbcd_budget_0p05_seed_0.npz'
# path_1 = 'examples/graph/arxiv_pertubated_adj/features_ori.pt'
# path_2 = 'examples/graph/arxiv_pertubated_adj/adj_ori.pt'
# path_3 = 'examples/graph/arxiv_pertubated_adj/lables_ori.pt'
# path_4 = 'examples/graph/arxiv_pertubated_adj/idx_train_ori.pt'
# path_5 = 'examples/graph/arxiv_pertubated_adj/idx_val_ori.pt'
# path_6 = 'examples/graph/arxiv_pertubated_adj/idx_test_ori.pt'

path_1 = 'examples/graph/arxiv/features.pt'
path_2 = 'examples/graph/arxiv/ogbn_arxiv_greedyrbcd_budget_0p01_seed_1.npz'
# path_2 = 'examples/graph/arxiv/ogbn_arxiv_prbcd_budget_0p1_seed_5.npz'
path_3 = 'examples/graph/arxiv/lables.pt'
path_4 = 'examples/graph/arxiv/idx_train.pt'
path_5 = 'examples/graph/arxiv/idx_val.pt'
path_6 = 'examples/graph/arxiv/idx_test.pt'

features, adj, labels = torch.load(path_1), torch.load(path_2), torch.load(path_3)
idx_train, idx_val, idx_test = torch.load(path_4), torch.load(path_5), torch.load(path_6)

adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
adj[adj>1] = 1

model = GCN(nfeat=features.shape[1], nhid=16, nclass=labels.max()+1, device=device)
model = model.to(device)

model.fit(features, adj, labels, idx_train, idx_val, train_iters=300, lip_norm=True, lip_node=False, lip_relu=False, gamma=0.0002, verbose=True)
model.eval()
model.test(idx_test)



# from scipy import sparse
# from ogb.nodeproppred import PygNodePropPredDataset
# import torch
# import numpy as np
# import torch.nn.functional as F
# import torch.optim as optim
# from deeprobust.graph.defense.gcn_lip import GCN
# from deeprobust.graph.global_attack import DICE
# from deeprobust.graph.defense.gcn_preprocess_lip import GCNJaccard
# from deeprobust.graph.defense.gcn_preprocess_lip import GCNSVD
# from deeprobust.graph.utils import *
# from deeprobust.graph.data import Dataset
# import scipy
# import argparse
# import os
# import numpy as np
#
# parser = argparse.ArgumentParser()
# parser.add_argument('--seed', type=int, default=15, help='Random seed.')
# parser.add_argument('--dataset', type=str, default='arxiv', choices=['cora', 'cora_ml', 'citeseer', 'polblogs', 'pubmed'], help='dataset')
# parser.add_argument('--ptb_rate', type=float, default=0.20,  help='pertubation rate')
#
# args = parser.parse_args()
# args.cuda = torch.cuda.is_available()
# print('cuda: %s' % args.cuda)
# device = torch.device("cuda:7" if torch.cuda.is_available() else "cpu")
#
# np.random.seed(args.seed)
# torch.manual_seed(args.seed)
# if args.cuda:
#     torch.cuda.manual_seed(args.seed)
#
# def edge_index_to_coo_dense(edge_index, N):
#     v = torch.ones(edge_index.size(1))
#     adj_coo = torch.sparse_coo_tensor(edge_index, v, (N, N))
#     return adj_coo.to_dense()
#
#
#
# # dataset = PygNodePropPredDataset(name = 'ogbn-arxiv')
# # data = dataset[0]
# # adj, features, labels = data.edge_index, data.x, data.y
# # split_idx = dataset.get_idx_split()
# # train_idx, val_idx, test_idx = split_idx['train'], split_idx['valid'], split_idx['test']
# path_1 = 'examples/graph/arxiv_pertubated_adj/features_ori.pt'
# path_2 = 'examples/graph/arxiv_pertubated_adj/adj_ori.pt'
# path_3 = 'examples/graph/arxiv_pertubated_adj/lables_ori.pt'
# path_4 = 'examples/graph/arxiv_pertubated_adj/idx_train_ori.pt'
# path_5 = 'examples/graph/arxiv_pertubated_adj/idx_val_ori.pt'
# path_6 = 'examples/graph/arxiv_pertubated_adj/idx_test_ori.pt'
#
# features, adj, labels = torch.load(path_1), torch.load(path_2), torch.load(path_3)
# idx_train, idx_val, idx_test = torch.load(path_4), torch.load(path_5), torch.load(path_6)
#
# edge_set = {(u.item(), v.item()) if u < v else (v.item(), u.item()) for u, v in  adj.T}
# pert = np.load('examples/graph/arxiv_pertubated_adj/ogbn_arxiv_greedyrbcd_budget_0p1_seed_0.npz')
# pert_removed_set = {(u, v) for u, v in pert['pert_removed'].T}
# pert_added_set = {(u, v) for u, v in pert['pert_added'].T}
# pert_edge_set = edge_set - pert_removed_set | pert_added_set
# pert_edge_index = torch.tensor(list(pert_edge_set)).T
#
# adj = pert_edge_index
# idx_train, idx_val, idx_test = idx_train.numpy(), idx_val.numpy(), idx_test.numpy()
# adj = edge_index_to_coo_dense(adj, features.shape[0])
# adj, features, labels = sparse.csr_matrix(adj), sparse.csr_matrix(features), labels.numpy()
# labels = labels.squeeze(-1)
#
# modified_adj = adj
# path = 'examples/graph/arxiv/ogbn_arxiv_greedyrbcd_budget_0p1_seed_0.npz'
torch.save(modified_adj, path)


