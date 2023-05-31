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

parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=15, help='Random seed.')
parser.add_argument('--dataset', type=str, default='citeseer', choices=['cora', 'cora_ml', 'citeseer', 'polblogs', 'pubmed'], help='dataset')
parser.add_argument('--ptb_rate', type=float, default=0.25,  help='pertubation rate')

args = parser.parse_args()
args.cuda = torch.cuda.is_available()
print('cuda: %s' % args.cuda)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

## >>> from deeprobust.graph.data import Dataset
## >>> from deeprobust.graph.global_attack import DICE
## >>> data = Dataset(root='/tmp/', name='cora')
## >>> adj, features, labels = data.adj, data.features, data.labels
## >>> model = DICE()
## >>> model.attack(adj, labels, n_perturbations=10)
# >>> modified_adj = model.modified_adj

# data = Dataset(root='/tmp/', name=args.dataset)
data = Dataset(root='/tmp/', name=args.dataset)
adj, features, labels = data.adj, data.features, data.labels
idx_train, idx_val, idx_test = data.idx_train, data.idx_val, data.idx_test
idx_unlabeled = np.union1d(idx_val, idx_test)

# Setup Attack Model
# model = DICE()

# n_perturbations = int(args.ptb_rate * (adj.sum()//2))

# model.attack(adj, labels, n_perturbations)

adj, features, labels = preprocess(adj, features, labels, preprocess_adj=False, sparse=True, device=device)

# modified_adj = model.modified_adj

# modified_adj = normalize_adj(modified_adj)
# modified_adj = sparse_mx_to_torch_sparse_tensor(modified_adj)
# modified_adj = modified_adj.to(device)

# # print(adj)
# # print(modified_adj)
# #直接保存相关文件，避免冗余实验
#
path_1 = "examples/graph/modified_adj_update/citeseer_dice_normal_0.25.pt"
# path_2 = "examples/graph/modified_adj_update/citeseer_dice_attack_0.20.pt"
#
# # torch.save(adj, path_1)
# # torch.save(modified_adj, path_2)
# original_adj = torch.load(path_1)
modified_adj = torch.load(path_1)



def test(adj):
    ''' test on GCN '''
    # adj = normalize_adj_tensor(adj)
    gcn = GCN(nfeat=features.shape[1],
              nhid=16,
              nclass=labels.max().item() + 1,
              dropout=0.5, device=device)

    gcn = gcn.to(device)

    # optimizer = optim.Adam(gcn.parameters(),
    #                        lr=0.01, weight_decay=5e-4)
    # gcn.fit(features, adj, labels, idx_train) # train without model picking
    # gcn.fit(features, adj, labels, idx_train, idx_val) # train with validation model picking
    gcn.fit(features, adj, labels, idx_train, idx_val, train_iters=200, lip_norm=False, lip_node=False, lip_relu=True, gamma=0.01, verbose=True)   #引入Lipschitz算法后
    output = gcn.output
    loss_test = F.nll_loss(output[idx_test], labels[idx_test])
    acc_test = accuracy(output[idx_test], labels[idx_test])
    print("Test set results:",
          "loss= {:.4f}".format(loss_test.item()),
          "accuracy= {:.4f}".format(acc_test.item()))

    return acc_test.item()

def main():
    # print('=== testing GCN on original(clean) graph ===')
    # test(adj)
    print('=== testing GCN on perturbed graph ===')
    test(modified_adj)
    # print(modified_adj)

if __name__ == '__main__':
    main()

