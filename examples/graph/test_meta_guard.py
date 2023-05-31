import torch
import numpy as np
import torch.nn.functional as F
# from deeprobust.graph.defense import GCN
from deeprobust.graph.utils import *
from deeprobust.graph.data import Dataset
from deeprobust.graph.data import PtbDataset, PrePtbDataset
from deeprobust.graph.defense.gcn_lip_guard import GCN
import argparse
from scipy import sparse

parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=15, help='Random seed.')
parser.add_argument('--dataset', type=str, default='citeseer', choices=['cora', 'cora_ml', 'citeseer', 'polblogs', 'pubmed'], help='dataset')
parser.add_argument('--ptb_rate', type=float, default=0.15,  help='pertubation rate')
parser.add_argument('--modelname', type=str, default='GCN',  choices=['GCN', 'GAT','GIN', 'JK'])
parser.add_argument('--defensemodel', type=str, default='GCNJaccard',  choices=['GCNJaccard', 'RGCN', 'GCNSVD'])
parser.add_argument('--GNNGuard', type=bool, default=True,  choices=[True, False])

args = parser.parse_args()
args.cuda = torch.cuda.is_available()
print('cuda: %s' % args.cuda)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Here the random seed is to split the train/val/test data,
# we need to set the random seed to be the same as that when you generate the perturbed graph
# data = Dataset(root='/tmp/', name=args.dataset, setting='prognn', seed=args.seed)
# Or we can just use setting='prognn' to get the splits
data = Dataset(root='/tmp/', name=args.dataset, setting='prognn')
adj, features, labels = data.adj, data.features, data.labels
idx_train, idx_val, idx_test = data.idx_train, data.idx_val, data.idx_test


print('==================')
print('=== load graph perturbed by Zugner metattack (under prognn splits) ===')
perturbed_data = PrePtbDataset(root='/tmp/',                # PrePtbDataset : meta-attack源码提供的pre-perturbed数据，只有ptb_rate=0.05-0.25
        name=args.dataset,
        attack_method='meta',
        ptb_rate=args.ptb_rate)

# perturbed_adj = adj      # 没有受到攻击时的邻接矩阵
perturbed_adj = perturbed_data.adj
modified_adj = torch.tensor(perturbed_adj.todense())
modified_adj = modified_adj.cpu().numpy()  # tensor->ndarray->scipy.csr.csr_matrix
modified_adj = sparse.csr_matrix(modified_adj)  # 转化成所需的输入形式

np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

attention = args.GNNGuard # if True, our method; if False, run baselines

def test(adj):
    # """defense models"""
    # classifier = GCNJaccard(nnodes=adj.shape[0], nfeat=features.shape[1], nhid=16,
    #                                           nclass=labels.max().item() + 1, dropout=0.5, device=device)
    # classifier = GCNSVD(nnodes=adj.shape[0], nfeat=features.shape[1], nhid=16,
    #                                           nclass=labels.max().item() + 1, dropout=0.5, device=device)

    # classifier = globals()[args.modelname](nfeat=features.shape[1], nhid=16, nclass=labels.max().item() + 1,
    #                                        dropout=0.5, device=device)
    ''' testing model '''
    classifier =  GCN(nfeat=features.shape[1], nclass=labels.max().item() + 1, nhid=16,
               dropout=0.5, with_relu=False, with_bias=True, weight_decay=5e-4,device=device)

    #
    classifier = classifier.to(device)

    classifier.fit(features, adj, labels, idx_train, train_iters=201,
                   idx_val=idx_val,
                   idx_test=idx_test,
                   verbose=True, attention=attention) # idx_val=idx_val, idx_test=idx_test , model_name=model_name
    classifier.eval()

    # classifier.fit(features, adj, labels, idx_train, idx_val) # train with validation model picking
    # acc_test, output = classifier.test(idx_test)
    acc_test = classifier.test(idx_test)
    return acc_test

def main():


    # print('=== testing GCN on original(clean) graph ===')
    # test(adj)

    print('=== testing GCN on Mettacked graph ===')
    test(modified_adj)

if __name__ == '__main__':
    main()

