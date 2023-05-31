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

# edge_set = {(u.item(), v.item()) if u < v else (v.item(), u.item()) for u, v in  adj.T}
# pert = np.load('examples/graph/arxiv_pertubated_adj/ogbn_arxiv_prbcd_budget_0p1_seed_1.npz')
# pert_removed_set = {(u, v) for u, v in pert['pert_removed'].T}
# pert_added_set = {(u, v) for u, v in pert['pert_added'].T}
# pert_edge_set = edge_set - pert_removed_set | pert_added_set
# pert_edge_index = torch.tensor(list(pert_edge_set)).T

# split_idx = dataset.get_idx_split()
#
# train_idx, val_idx, test_idx = split_idx['train'], split_idx['valid'], split_idx['test']
# idx_train, idx_val, idx_test = train_idx.numpy(), val_idx.numpy(), test_idx.numpy()
# adj = edge_index_to_coo_dense(adj, features.shape[0])
# adj, features, labels = sparse.csr_matrix(adj), sparse.csr_matrix(features), labels.numpy()
# labels = labels.squeeze(-1)
# path_1 = 'examples/graph/arxiv/features.pt'
# path_2 = 'examples/graph/arxiv/adj.pt'
# path_3 = 'examples/graph/arxiv/lables.pt'
# path_4 = 'examples/graph/arxiv/idx_train.pt'
# path_5 = 'examples/graph/arxiv/idx_val.pt'
# path_6 = 'examples/graph/arxiv/idx_test.pt'

path_1 = 'examples/graph/arxiv_pertubated_adj/features_ori.pt'
path_2 = 'examples/graph/arxiv_pertubated_adj/adj_ori.pt'
path_3 = 'examples/graph/arxiv_pertubated_adj/lables_ori.pt'
path_4 = 'examples/graph/arxiv_pertubated_adj/idx_train_ori.pt'
path_5 = 'examples/graph/arxiv_pertubated_adj/idx_val_ori.pt'
path_6 = 'examples/graph/arxiv_pertubated_adj/idx_test_ori.pt'
#
torch.save(features, path_1)
torch.save(adj, path_2)
torch.save(labels, path_3)
torch.save(train_idx, path_4)
torch.save(val_idx, path_5)
torch.save(test_idx, path_6)

features, adj, labels = torch.load(path_1), torch.load(path_2), torch.load(path_3)
idx_train, idx_val, idx_test = torch.load(path_4), torch.load(path_5), torch.load(path_6)


adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
adj[adj>1] = 1
model = GCN(nfeat=features.shape[1], nhid=32, nclass=labels.max()+1, device=device)
model = model.to(device)

# model.fit(features, perturbed_adj, labels, idx_train, train_iters=200, verbose=True)
# using validation to pick model
model.fit(features, adj, labels, idx_train, idx_val, train_iters=300, lip_norm=False, lip_node=False, lip_relu=False, gamma=0.00001, verbose=True)
model.eval()
# You can use the inner function of model to test
model.test(idx_test)

# # Setup Attack Model
# model = DICE()
#
# n_perturbations = int(args.ptb_rate * (adj.sum()//2))
#
# model.attack(adj, labels, n_perturbations)
# #
# adj, features, labels = preprocess(adj, features, labels, preprocess_adj=False, sparse=True, device=device)
#
#
# modified_adj = model.modified_adj
#
# modified_adj = normalize_adj(modified_adj)
# modified_adj = sparse_mx_to_torch_sparse_tensor(modified_adj)
# modified_adj = modified_adj.to(device)

# # print(adj)
# # print(modified_adj)
# #直接保存相关文件，避免冗余实验
#
# path_1 = "examples/graph/modified_adj_update/cora_dice_normal_0.05.pt"
# path_2 = "examples/graph/modified_adj_update/citeseer_dice_attack_0.25.pt"
#
# # torch.save(adj, path_1)
# # torch.save(modified_adj, path_2)
# original_adj = torch.load(path_1)
# modified_adj = torch.load(path_2)



def test(adj):
    ''' test on GCN '''
    # adj = normalize_adj_tensor(adj)
    gcn = GCN(nfeat=features.shape[1],
              nhid=16,
              nclass=labels.max().item() + 1,
              dropout=0.5, device=device)

    gcn = gcn.to(device)

    # gcn.fit(features, adj, labels, idx_train, idx_val) # train with validation model picking
    gcn.fit(features, adj, labels, idx_train, idx_val, train_iters=200, lip_norm=False, lip_node=False, lip_relu=False, gamma=0.001, verbose=True)   #引入Lipschitz算法后
    output = gcn.output
    loss_test = F.nll_loss(output[idx_test], labels[idx_test])
    acc_test = accuracy(output[idx_test], labels[idx_test])
    print("Test set results:",
          "loss= {:.4f}".format(loss_test.item()),
          "accuracy= {:.4f}".format(acc_test.item()))

    return acc_test.item()


# def main():
    # print('=== testing GCN on original(clean) graph ===')
    # test(adj)
    # print('=== testing GCN on perturbed graph ===')
    # test(modified_adj)

# if __name__ == '__main__':
#     main()
