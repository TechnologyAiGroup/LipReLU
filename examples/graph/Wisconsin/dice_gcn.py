import torch
import argparse
from deeprobust.graph.defense.gcn_lip import GCN
from deeprobust.graph.global_attack import DICE
from deeprobust.graph.utils import *
import random
from scipy import sparse
from examples.graph.OtherDataset.loader import load_new_data
from deeprobust.graph.defense.gcn_preprocess_lip import GCNSVD
from deeprobust.graph.defense.gcn_preprocess_lip import GCNJaccard
from deeprobust.graph.defense.r_gcn_lip import RGCN

parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=10, help='Random seed.')
parser.add_argument('--dataset', type=str, default='wisconsin', choices=['chameleon', 'cormelon', 'film', 'squirrel', 'texas', 'wisconsin'], help='dataset')
parser.add_argument('--ptb_rate', type=float, default=0.25,  help='perturbation rate')   # 0.01 0.11 0.15 0.20 0.25

parser.add_argument('--lr', type=float, default=0.01, help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=5e-4, help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden', type=int, default=16, help='Number of hidden units.')
parser.add_argument('--dropout', type=float, default=0.5, help='Dropout rate (1 - keep probability).')

args = parser.parse_args()
args.cuda = torch.cuda.is_available()
print('cuda: %s' % args.cuda)
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
perturbations = int(args.ptb_rate * (adj.sum()//2))

# # Setup Attack Model
# model = DICE()
# n_perturbations = int(args.ptb_rate * (adj.sum()//2))
# model.attack(adj, labels, n_perturbations)
# modified_adj = model.modified_adj
#
path_1 = "examples/graph/OtherDataset/dice_modified_adj/wisconsin_normal.pt"
path_2 = "examples/graph/OtherDataset/dice_modified_adj/wisconsin_ptb_0.10.pt"
#
# torch.save(adj, path_1)
# torch.save(modified_adj, path_2)
adj = torch.load(path_1)

modified_adj = torch.load(path_2)


def test(features, adj, labels):
    ''' test on GCN '''

    adj, features, labels = preprocess(adj, features, labels, preprocess_adj=False, sparse=True, device=device)

    gcn = GCN(nfeat=features.shape[1],
              nhid=16,
              nclass=labels.max().item() + 1,
              dropout=0.5, device=device)

    gcn = gcn.to(device)
    gcn.fit(features, adj, labels, idx_train, idx_val, train_iters=200, lip_norm=False, lip_node=False, lip_relu=False, gamma=0.001, verbose=True)   #引入Lipschitz算法后
    output = gcn.output
    loss_test = F.nll_loss(output[idx_test], labels[idx_test])
    acc_test = accuracy(output[idx_test], labels[idx_test])
    print("Test set results:",
          "loss= {:.4f}".format(loss_test.item()),
          "accuracy= {:.4f}".format(acc_test.item()))

    return acc_test.item()

def test_Jaccard(adj):
    ''' test on GCN '''
    # adj = normalize_adj_tensor(adj)
    gcn = GCNJaccard(nfeat=features.shape[1],
              nhid=16,
              nclass=labels.max().item() + 1,
              dropout=0.5, device=device)

    gcn = gcn.to(device)

    gcn.fit(features, adj, labels, idx_train, idx_val, threshold=0.01, lip_norm=False, lip_node=False, lip_relu=True, gamma=0.045, verbose=True)
    gcn.eval()
    gcn.test(idx_test)
#
def test_SVD(adj):
    ''' test on GCN '''
    # adj = normalize_adj_tensor(adj)
    gcn = GCNSVD(nfeat=features.shape[1], nclass=labels.max() + 1,
                   nhid=16, device=device)

    gcn = gcn.to(device)

    gcn.fit(features, adj, labels, idx_train, idx_val, k=30, lip_norm=False, lip_node=False, lip_relu=True, gamma=0.037, verbose=False)
    gcn.eval()
    gcn.test(idx_test)

def test_RGCN(adj):
    ''' test on GCN '''
    # adj = normalize_adj_tensor(adj)
    gcn = RGCN(nnodes=adj.shape[0], nfeat=features.shape[1], nclass=labels.max()+1,
                nhid=32, device=device)

    gcn = gcn.to(device)

    gcn.fit(features, adj, labels, idx_train, idx_val, train_iters=200, lip_norm=True, gamma=0.025, verbose=False)
    gcn.eval()
    gcn.test(idx_test)


def main():

    # print('=== testing GCN on original(clean) graph ===')
    # test(features, adj, labels)
    print('=== testing GCN on modified(noisy) graph ===')
    test(features, modified_adj, labels)
    # print('=== testing GCN on original(clean) graph ===')
    # test_Jaccard(adj)
    # print('=== testing GCN on modified(noisy) graph ===')
    # test_Jaccard(modified_adj)
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