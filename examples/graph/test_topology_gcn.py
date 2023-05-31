import torch
import numpy as np
import torch.nn.functional as F
import torch.optim as optim
# from deeprobust.graph.defense import GCN
from scipy import sparse
from deeprobust.graph.defense.gcn_lip import GCN
from deeprobust.graph.global_attack import MinMax
from deeprobust.graph.utils import *
from deeprobust.graph.data import Dataset
from deeprobust.graph.utils import preprocess
from deeprobust.graph.defense.gcn_preprocess_lip import GCNJaccard
from deeprobust.graph.defense.gcn_preprocess_lip import GCNSVD
from deeprobust.graph.defense.r_gcn_lip import RGCN
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=15, help='Random seed.')
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
parser.add_argument('--ptb_rate', type=float, default=0.05,  help='pertubation rate')
# parser.add_argument('--model', type=str, default='PGD', choices=['PGD', 'min-max'], help='model variant')

args = parser.parse_args()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# device = 'cpu'

np.random.seed(args.seed)
torch.manual_seed(args.seed)
if device != 'cpu':
    torch.cuda.manual_seed(args.seed)

data = Dataset(root='/tmp/', name=args.dataset)
adj, features, labels = data.adj, data.features, data.labels
idx_train, idx_val, idx_test = data.idx_train, data.idx_val, data.idx_test
# idx_unlabeled = np.union1d(idx_val, idx_test)

perturbations = int(args.ptb_rate * (adj.sum()//2))

# # Setup Victim Model
# def attack_opology(features, adj, labels):
#     victim_model = GCN(nfeat=features.shape[1], nclass=labels.max().item() + 1, nhid=16,
#                        dropout=0.5, weight_decay=5e-4, device=device)
#     #
#     victim_model = victim_model.to(device)
#     victim_model.fit(features, adj, labels, idx_train)
#     #
#     # # Setup Attack Model
#     #
#     model = MinMax(model=victim_model, nnodes=adj.shape[0], loss_type='CE', device=device)
#     model = model.to(device)
#     model.attack(features, adj, labels, idx_train, perturbations)
#     modified_adj = model.modified_adj
#     return modified_adj
#
# adj_s, features_s, labels_s = preprocess(adj, features, labels, preprocess_adj=False)
# modified_adj = attack_opology(features_s, adj_s, labels_s)
#
#
path_1 = "examples/graph/topology_modified_adj/citeseer_topology_lip_adj_normal_ptb_rate_0.05.pt"
path_2 = "examples/graph/topology_modified_adj/citeseer_topology_lip_modified_adj_ptb_rate_0.05.pt"
#
# torch.save(adj, path_1)
# torch.save(modified_adj, path_2)
original_adj = torch.load(path_1)
modified_adj = torch.load(path_2)
#
modified_adj_array = modified_adj.cpu().numpy()               # tensor->ndarray->scipy.csr.csr_matrix
modified_adj_scipy = sparse.csr_matrix(modified_adj_array)    #转化成所需的输入形式

def test_Jaccard(adj):
    ''' test on GCN '''
    # adj = normalize_adj_tensor(adj)
    gcn = GCNJaccard(nfeat=features.shape[1],
              nhid=16,
              nclass=labels.max().item() + 1,
              dropout=0.5, device=device)

    gcn = gcn.to(device)

    gcn.fit(features, adj, labels, idx_train, idx_val, threshold=0.02, lip_norm=False, lip_node=False, lip_relu=False, gamma=0.035, verbose=True)
    # gcn.fit(features, adj, labels, idx_train, threshold=0.01, lip_norm=False, gamma=0.012, verbose=False)
    gcn.eval()
    gcn.test(idx_test)

# def test_SVD(adj):
#     ''' test on GCN '''
#     # adj = normalize_adj_tensor(adj)
#     gcn = GCNSVD(nfeat=features.shape[1], nclass=labels.max() + 1,
#                    nhid=16, device=device)
#
#     gcn = gcn.to(device)
#
#     gcn.fit(features, adj, labels, idx_train, idx_val, k=50, lip_norm=True, lip_node=True, lip_relu=True,  gamma=0.005, verbose=True)
#     # gcn.fit(features, adj, labels, idx_train, k=50, lip_norm=True, gamma=0.00006, verbose=False)
#     gcn.eval()
#     gcn.test(idx_test)
# #
# def test_RGCN(adj):
#     ''' test on GCN '''
#     # adj = normalize_adj_tensor(adj)
#     gcn = RGCN(nnodes=adj.shape[0], nfeat=features.shape[1], nclass=labels.max()+1,
#                 nhid=32, device=device)
#
#     gcn = gcn.to(device)
#
#     gcn.fit(features, adj, labels, idx_train, idx_val, train_iters=200, lip_norm=True, gamma=0.0095, verbose=False)
#     # gcn.fit(features, adj, labels, idx_train, train_iters=200, lip_norm=False, gamma=0.025, verbose=False)
#     gcn.eval()
#     gcn.test(idx_test)


# def test(adj):
#     ''' test on GCN '''
#
#     adj = normalize_adj_tensor(adj)
#     gcn = GCN(nfeat=features.shape[1],
#               nhid=args.hidden,
#               nclass=labels.max().item() + 1,
#               dropout=args.dropout, device=device)
#     gcn = gcn.to(device)
#     gcn.fit(features, adj, labels, idx_train, train_iters=200, lip_norm=False, gamma=0.1) # train without model picking
#     # gcn.fit(features, adj, labels, idx_train, idx_val)                                  # train with validation model picking
#     output = gcn.output.to('cpu')
#     loss_test = F.nll_loss(output[idx_test], labels[idx_test])
#     acc_test = accuracy(output[idx_test], labels[idx_test])
#     print("Test set results:",
#           "loss= {:.4f}".format(loss_test.item()),
#           "accuracy= {:.4f}".format(acc_test.item()))
#
#     return acc_test.item()


def main():

    # print('=== testing GCN on original(clean) graph ===')
    # test(original_adj)
    # print('=== testing GCN on modified(noisy) graph ===')
    # test(modified_adj_scipy)
    print('=== testing GCN on original(clean) graph ===')
    test_Jaccard(original_adj)
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

