import torch
import numpy as np
import torch.nn.functional as F
# from deeprobust.graph.defense import GCN
from deeprobust.graph.defense.gcn_lip import GCN
from deeprobust.graph.utils import *
from deeprobust.graph.data import Dataset
from deeprobust.graph.data import PtbDataset, PrePtbDataset
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=15, help='Random seed.')
# parser.add_argument('--seed', type=int, default=50, help='Random seed.')
parser.add_argument('--dataset', type=str, default='cora', choices=['cora', 'cora_ml', 'citeseer', 'polblogs', 'pubmed'], help='dataset')
parser.add_argument('--ptb_rate', type=float, default=0.15,  help='pertubation rate')

args = parser.parse_args()
args.cuda = torch.cuda.is_available()
print('cuda: %s' % args.cuda)
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

# Here the random seed is to split the train/val/test data,
# we need to set the random seed to be the same as that when you generate the perturbed graph
# data = Dataset(root='/tmp/', name=args.dataset, setting='prognn', seed=args.seed)
# Or we can just use setting='prognn' to get the splits
data = Dataset(root='/tmp/', name=args.dataset, setting='prognn')
adj, features, labels = data.adj, data.features, data.labels
idx_train, idx_val, idx_test = data.idx_train, data.idx_val, data.idx_test


# load pre-attacked graph by Zugner: https://github.com/danielzuegner/gnn-meta-attack
print('==================')
print('=== load graph perturbed by Zugner metattack (under prognn splits) ===')
perturbed_data = PrePtbDataset(root='/tmp/',                # PrePtbDataset : meta-attack源码提供的pre-perturbed数据，只有ptb_rate=0.05-0.25
        name=args.dataset,
        attack_method='meta',
        ptb_rate=args.ptb_rate)

# perturbed_adj = adj      # 没有受到攻击时的邻接矩阵
perturbed_adj = perturbed_data.adj
per_adj_dense = torch.tensor(perturbed_adj.todense())

# perturbed_adj = sp.load_npz("examples/graph/meta_attack_adj/cora_meta_adj_0.25.npz") # cora ptb=0.25

np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

# Setup GCN Model

model = GCN(nfeat=features.shape[1], nhid=16, nclass=labels.max()+1, device=device)
model = model.to(device)


# model.fit(features, perturbed_adj, labels, idx_train, train_iters=200, verbose=True)
# # using validation to pick model
model.fit(features, perturbed_adj, labels, idx_train, idx_val, train_iters=200, lip_norm=False, lip_node=False, lip_relu=False, gamma=0.001, verbose=True)
model.eval()
# You can use the inner function of model to test
model.test(idx_test)

# print('==================')
# print('=== load graph perturbed by DeepRobust 5% metattack (under prognn splits) ===')
# perturbed_data = PtbDataset(root='/tmp/',   # PtbDataset : pro-gnn源码提供的pre-perturbed数据，ptb_rate=0.05
#                     name=args.dataset,
#                     attack_method='meta')
# perturbed_adj = perturbed_data.adj
#
# model.fit(features, perturbed_adj, labels, idx_train, train_iters=200, verbose=True)
# # # using validation to pick model
# # model.fit(features, perturbed_adj, labels, idx_train, idx_val, train_iters=200, verbose=True)
# model.eval()
# # You can use the inner function of model to test
# model.test(idx_test)

