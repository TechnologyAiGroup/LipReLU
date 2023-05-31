import torch
import numpy as np
import torch.nn.functional as F
import torch.optim as optim
from deeprobust.graph.defense.gcn_lip import GCN
from deeprobust.graph.global_attack import DICE
from deeprobust.graph.defense.gcn_preprocess_lip import GCNJaccard
from deeprobust.graph.defense.gcn_preprocess_lip import GCNSVD
from deeprobust.graph.defense.r_gcn_lip import RGCN
from deeprobust.graph.utils import *
from deeprobust.graph.data import Dataset
import scipy

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=15, help='Random seed.')
parser.add_argument('--dataset', type=str, default='cora', choices=['cora', 'cora_ml', 'citeseer', 'polblogs', 'pubmed'], help='dataset')
parser.add_argument('--ptb_rate', type=float, default=0.05,  help='pertubation rate')


args = parser.parse_args()
args.cuda = torch.cuda.is_available()
print('cuda: %s' % args.cuda)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

# >>> from deeprobust.graph.data import Dataset
# >>> from deeprobust.graph.global_attack import DICE
# >>> data = Dataset(root='/tmp/', name='cora')
# >>> adj, features, labels = data.adj, data.features, data.labels
# >>> model = DICE()
# >>> model.attack(adj, labels, n_perturbations=10)
# >>> modified_adj = model.modified_adj

# data = Dataset(root='/tmp/', name=args.dataset)
data = Dataset(root='/tmp/', name=args.dataset)
adj, features, labels = data.adj, data.features, data.labels
idx_train, idx_val, idx_test = data.idx_train, data.idx_val, data.idx_test
idx_unlabeled = np.union1d(idx_val, idx_test)

# # Setup Attack Model
# model = DICE()
# n_perturbations = int(args.ptb_rate * (adj.sum()//2))
# model.attack(adj, labels, n_perturbations)
#
# modified_adj = model.modified_adj
# original_adj = adj

# path_1 = "examples/graph/modified_adj/cora_dice_normal_jaccard_0.05.pt"
# path_2 = "examples/graph/modified_adj/cora_dice_attack_jaccard_0.15.pt"

# torch.save(original_adj, path_1)
# torch.save(modified_adj, path_2)
# original_adj = torch.load(path_1)
# modified_adj = torch.load(path_2)

# print(original_adj)
# print(modified_adj)
# print(type(original_adj))
# print(type(original_adj))

def test_Jaccard(adj):
    ''' test on GCN '''
    # adj = normalize_adj_tensor(adj)
    gcn = GCNJaccard(nfeat=features.shape[1],
              nhid=16,
              nclass=labels.max().item() + 1,
              dropout=0.5, device=device)

    gcn = gcn.to(device)

    gcn.fit(features, adj, labels, idx_train, idx_val, threshold=0.03, lip_norm=False, lip_node=False, lip_relu=False, gamma=0.045, verbose=True)
    gcn.eval()
    gcn.test(idx_test)
#
# def test_SVD(adj):
#     ''' test on GCN '''
#     # adj = normalize_adj_tensor(adj)
#     gcn = GCNSVD(nfeat=features.shape[1], nclass=labels.max() + 1,
#                    nhid=16, device=device)
#
#     gcn = gcn.to(device)
#
#     gcn.fit(features, adj, labels, idx_train, idx_val, k=30, lip_norm=False, lip_node=False, lip_relu=True, gamma=0.037, verbose=False)
#     gcn.eval()
#     gcn.test(idx_test)
# # #
# def test_RGCN(adj):
#     ''' test on GCN '''
#     # adj = normalize_adj_tensor(adj)
#     gcn = RGCN(nnodes=adj.shape[0], nfeat=features.shape[1], nclass=labels.max()+1,
#                 nhid=32, device=device)
#
#     gcn = gcn.to(device)
#
#     gcn.fit(features, adj, labels, idx_train, idx_val, train_iters=200, lip_norm=True, gamma=0.025, verbose=False)
#     gcn.eval()
#     gcn.test(idx_test)
#
def main():

    print('______________________________________________________')
    # test_Jaccard(original_adj)
    test_Jaccard(modified_adj)
    # print('______________________________________________________')
    # test_SVD(original_adj)
    # test_SVD(modified_adj)
    # print('______________________________________________________')
    # test_RGCN(original_adj)
    # test_RGCN(modified_adj)

if __name__ == '__main__':
    main()

