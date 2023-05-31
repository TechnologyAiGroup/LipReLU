import torch
import numpy as np
import torch.nn.functional as F
# from deeprobust.graph.defense import GCN
from deeprobust.graph.defense.gcn_lip import GCN
from deeprobust.graph.utils import *
from deeprobust.graph.data import Dataset
from deeprobust.graph.data import PtbDataset, PrePtbDataset
from deeprobust.graph.defense.gcn_preprocess_lip import GCNJaccard
from deeprobust.graph.defense.gcn_preprocess_lip import GCNSVD
from deeprobust.graph.defense.r_gcn_lip import RGCN

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=15, help='Random seed.')
# parser.add_argument('--seed', type=int, default=50, help='Random seed.')
parser.add_argument('--dataset', type=str, default='cora', choices=['cora', 'cora_ml', 'citeseer', 'polblogs', 'pubmed'], help='dataset')
parser.add_argument('--ptb_rate', type=float, default=0.15,  help='pertubation rate')
parser.add_argument('--k', type=int, default=45, help='Truncated Components.')

args = parser.parse_args()
args.cuda = torch.cuda.is_available()
print('cuda: %s' % args.cuda)
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

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
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

def test(perturbed_adj):

    model = GCN(nfeat=features.shape[1], nhid=16, nclass=labels.max() + 1, device=device)
    model = model.to(device)

    # # using validation to pick model
    model.fit(features, perturbed_adj, labels, idx_train, idx_val, train_iters=200, lip_norm=False, lip_node=False,
              lip_relu=False, gamma=0.001, verbose=True)
    model.eval()
    # You can use the inner function of model to test
    model.test(idx_test)

def test_jaccard(perturbed_adj):

    model = GCNJaccard(nfeat=features.shape[1], nclass=labels.max() + 1,
                       nhid=16, device=device)
    model = model.to(device)
    model.fit(features, perturbed_adj, labels, idx_train, idx_val, threshold=0.03, lip_norm=False, lip_node=False,
              lip_relu=False, gamma=0.01, verbose=True)
    model.eval()
    model.test(idx_test)

def test_svd(perturbed_adj):
    model = GCNSVD(nfeat=features.shape[1], nclass=labels.max() + 1,
                   nhid=16, device=device)

    model = model.to(device)

    print("perturbed_adj:", type(perturbed_adj))
    print('=== testing GCN-SVD on perturbed graph ===')
    model.fit(features, perturbed_adj, labels, idx_train, idx_val, k=args.k, lip_norm=False, lip_node=False,
              lip_relu=False, gamma=0.1, verbose=True)
    model.eval()
    model.test(idx_test)

def test_rgcn(perturbed_adj):
    # Setup RGCN Model
    model = RGCN(nnodes=perturbed_adj.shape[0], nfeat=features.shape[1], nclass=labels.max() + 1,
                 nhid=32, device=device)

    print("perturbed_adj:", type(perturbed_adj))
    model = model.to(device)

    model.fit(features, perturbed_adj, labels, idx_train, idx_val, train_iters=200, lip_norm=False, lip_node=False,
              gamma=0.0054, verbose=True)
    # You can use the inner function of model to test
    model.test(idx_test)




def main():

    test = test_template
    adj = perturbed_adj
    test(adj)
if __name__ == '__main__':
    main()



