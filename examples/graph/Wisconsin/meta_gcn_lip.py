import torch
import numpy as np
import torch.nn.functional as F
import torch.optim as optim
import random
from deeprobust.graph.defense import GCN
from deeprobust.graph.global_attack import MetaApprox, Metattack
from deeprobust.graph.utils import *
from examples.graph.OtherDataset.loader import load_new_data
from scipy import sparse
from deeprobust.graph.defense.gcn_preprocess_lip import GCNSVD
from deeprobust.graph.defense.gcn_preprocess_lip import GCNJaccard
from deeprobust.graph.defense.r_gcn_lip import RGCN
from deeprobust.graph.data import Dataset
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='Disables CUDA training.')

parser.add_argument('--seed', type=int, default=10, help='Random seed.')

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
parser.add_argument('--dataset', type=str, default='wisconsin', choices=['chameleon', 'cormelon', 'film', 'squirrel', 'texas', 'wisconsin'], help='dataset')
parser.add_argument('--ptb_rate', type=float, default=0.09,  help='pertubation rate')  # [seed=10 0.03 0.10 0.11 0.22 0.28 ]

parser.add_argument('--model', type=str, default='Meta-Self',
        choices=['Meta-Self', 'A-Meta-Self', 'Meta-Train', 'A-Meta-Train'], help='model variant')

args = parser.parse_args()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

def get_data_split():

    idx_train = np.array(range(60))
    idx_val = np.array(range(60,130))
    idx_test = np.array(range(130, 251))
    return idx_train, idx_val, idx_test

adj, features, labels = load_new_data(args.dataset, 'examples/graph/OtherDataset/data/wisconsin')
idx_train, idx_val, idx_test = get_data_split()
idx_unlabeled = np.union1d(idx_val, idx_test)
perturbations = int(args.ptb_rate * (adj.sum()//2))

def attack_opology(features, adj, labels):

    adj, features, labels = preprocess(adj, features, labels, preprocess_adj=False)
    # Setup Surrogate Model
    surrogate = GCN(nfeat=features.shape[1], nclass=labels.max().item() + 1, nhid=16,
                    dropout=0.5, with_relu=False, with_bias=True, weight_decay=5e-4, device=device)

    surrogate = surrogate.to(device)
    surrogate.fit(features, adj, labels, idx_train, idx_val)

    # Setup Attack Model
    if 'Self' in args.model:
        lambda_ = 0
    if 'Train' in args.model:
        lambda_ = 1
    if 'Both' in args.model:
        lambda_ = 0.5

    if 'A' in args.model:
        model = MetaApprox(model=surrogate, nnodes=adj.shape[0], feature_shape=features.shape, attack_structure=True,
                           attack_features=False, device=device, lambda_=lambda_)

    else:
        model = Metattack(model=surrogate, nnodes=adj.shape[0], feature_shape=features.shape, attack_structure=True,
                          attack_features=False, device=device, lambda_=lambda_)

    model = model.to(device)
    model.attack(features, adj, labels, idx_train, idx_unlabeled, perturbations, ll_constraint=False)
    modified_adj = model.modified_adj
    return modified_adj

# modified_adj = attack_opology(features, adj, labels)       # 在没有扰动时，先给注销掉，避免随机种子的影响，使得结果可以复现
#
path_1 = "examples/graph/OtherDataset/meta_modified_adj/wisconsin_normal.pt"
path_2 = "examples/graph/OtherDataset/meta_modified_adj/wisconsin_ptb_0.25.pt"


# original_adj = torch.load(path_1)
modified_adj = torch.load(path_2)

modified_adj_array = modified_adj.cpu().numpy()               # tensor->ndarray->scipy.csr.csr_matrix
modified_adj_scipy = sparse.csr_matrix(modified_adj_array)    #转化成所需的输入形式

# def test_Jaccard(adj):
#     ''' test on GCN '''
#     # adj = normalize_adj_tensor(adj)
#     gcn = GCNJaccard(nfeat=features.shape[1],
#               nhid=16,
#               nclass=labels.max().item() + 1,
#               dropout=0.5, device=device)
#
#     gcn = gcn.to(device)
#
#     gcn.fit(features, adj, labels, idx_train, idx_val, threshold=0.02, lip_norm=False, lip_node=False, lip_relu=True, gamma=0.035, verbose=True)
#     # gcn.fit(features, adj, labels, idx_train, threshold=0.01, lip_norm=False, gamma=0.012, verbose=False)
#     gcn.eval()
#     gcn.test(idx_test)

# def test_SVD(adj):
#     ''' test on GCN '''
#     # adj = normalize_adj_tensor(adj)
#     gcn = GCNSVD(nfeat=features.shape[1], nclass=labels.max() + 1,
#                    nhid=16, device=device)
#
#     gcn = gcn.to(device)
#
#     gcn.fit(features, adj, labels, idx_train, idx_val, k=50, lip_norm=False, lip_node=False, lip_relu=False,  gamma=0.005, verbose=True)
#     # gcn.fit(features, adj, labels, idx_train, k=50, lip_norm=True, gamma=0.00006, verbose=False)
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
#     gcn.fit(features, adj, labels, idx_train, idx_val, train_iters=200, lip_norm=False, gamma=0.0095, verbose=True)
#     # gcn.fit(features, adj, labels, idx_train, train_iters=200, lip_norm=False, gamma=0.025, verbose=False)
#     gcn.eval()
#     gcn.test(idx_test)

def test(features, adj, labels):
    ''' test on GCN '''

    # adj = normalize_adj_tensor(adj)
    adj, features, labels = preprocess(adj, features, labels, preprocess_adj = False)

    gcn = GCN(nfeat=features.shape[1],
              nhid=args.hidden,
              nclass=labels.max().item() + 1,
              dropout=args.dropout, device=device)
    gcn = gcn.to(device)
    gcn.fit(features, adj, labels, idx_train, idx_val, lip_node=True, lip_relu=False, lip_norm=False, verbose=True) # train without model picking
    # gcn.fit(features, adj, labels, idx_train, idx_val) # train with validation model picking
    output = gcn.output.cpu()
    loss_test = F.nll_loss(output[idx_test], labels[idx_test])
    acc_test = accuracy(output[idx_test], labels[idx_test])
    print("Test set results:",
          "loss= {:.4f}".format(loss_test.item()),
          "accuracy= {:.4f}".format(acc_test.item()))

    return acc_test.item()


def main():

    # print('=== testing GCN on original(clean) graph ===')
    # test(features, adj, labels)
    print('=== testing GCN on modified(noisy) graph ===')
    test(features, modified_adj_scipy, labels)
    # print('=== testing GCN on original(clean) graph ===')
    # test_Jaccard(adj)
    # print('=== testing GCN on modified(noisy) graph ===')
    # test_Jaccard(modified_adj_scipy)
    # print('=== testing GCN on original(clean) graph ===')
    # test_SVD(original_adj)
    # print('=== testing GCN on modified(noisy) graph ===')
    # test_SVD(modified_adj_scipy)
    # print('=== testing GCN on original(clean) graph ===')
    # test_RGCN(original_adj)
    # print('=== testing GCN on modified(noisy) graph ===')
    # test_RGCN(modified_adj_scipy)

if __name__ == '__main__':
    main()

