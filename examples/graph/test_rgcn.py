import torch
import numpy as np
import torch.nn.functional as F
# from deeprobust.graph.defense import RGCN
from deeprobust.graph.defense.r_gcn_lip import RGCN
from deeprobust.graph.utils import *
from deeprobust.graph.data import Dataset
from deeprobust.graph.data import PrePtbDataset
import random

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=15, help='Random seed.')
parser.add_argument('--dataset', type=str, default='cora', choices=['cora', 'cora_ml', 'citeseer', 'polblogs', 'pubmed'], help='dataset')
parser.add_argument('--ptb_rate', type=float, default=0.10,  help='pertubation rate')

args = parser.parse_args()
args.cuda = torch.cuda.is_available()
print('cuda: %s' % args.cuda)
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
# Here the random seed is to split the train/val/test data,
# we need to set the random seed to be the same as that when you generate the perturbed graph
# data = Dataset(root='/tmp/', name=args.dataset, setting='nettack', seed=15)
# Or we can just use setting='prognn' to get the splits
data = Dataset(root='/tmp/', name=args.dataset, setting='prognn')
adj, features, labels = data.adj, data.features, data.labels
idx_train, idx_val, idx_test = data.idx_train, data.idx_val, data.idx_test

np.random.seed(args.seed)
random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed_all(args.seed)
    torch.cuda.manual_seed(args.seed)

# load pre-attacked graph by Zugner: https://github.com/danielzuegner/gnn-meta-attack
print('==================')
print('=== load graph perturbed by Zugner metattack (under prognn splits) ===')

perturbed_data = PrePtbDataset(root='/tmp/',
        name=args.dataset,
        attack_method='meta',
        ptb_rate=args.ptb_rate)

# perturbed_adj = adj
perturbed_adj = perturbed_data.adj

# Setup RGCN Model
model = RGCN(nnodes=perturbed_adj.shape[0], nfeat=features.shape[1], nclass=labels.max()+1,
                nhid=32, device=device)

print("perturbed_adj:", type(perturbed_adj))
model = model.to(device)

model.fit(features, perturbed_adj, labels, idx_train, idx_val, train_iters=200, lip_norm=False, lip_node=False, gamma=0.0054, erbose=True)
# You can use the inner function of model to test
model.test(idx_test)

