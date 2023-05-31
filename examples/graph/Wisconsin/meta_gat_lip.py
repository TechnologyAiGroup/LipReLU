import torch
import argparse
import numpy as np
from deeprobust.graph.data import Dataset, Dpr2Pyg
# from deeprobust.graph.defense import GAT
from deeprobust.graph.defense.gat_lip_topo_pyg import GAT
from deeprobust.graph.global_attack import MetaApprox, Metattack
from examples.graph.OtherDataset.loader import load_new_data
from deeprobust.graph.data import Dataset
from deeprobust.graph.utils import preprocess
import random
from deeprobust.graph.defense.gcn_lip import GCN
from scipy import sparse
from deeprobust.graph.data import PrePtbDataset

parser = argparse.ArgumentParser()
# parser.add_argument('--seed', type=int, default=18, help='Random seed.')
parser.add_argument('--seed', type=int, default=15, help='Random seed.')
parser.add_argument('--dataset', type=str, default='wisconsin', choices=['chameleon', 'cormelon', 'film', 'squirrel', 'texas', 'wisconsin'], help='dataset')
parser.add_argument('--ptb_rate', type=float, default=0.10,  help='perturbation rate')

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
adj, features, labels = preprocess(adj, features, labels, preprocess_adj=False)
adj = adj.to(device)
features = features.to(device)
labels = labels.to(device)
idx_train, idx_val, idx_test = get_data_split()
idx_unlabeled = np.union1d(idx_val, idx_test)
perturbations = int(args.ptb_rate * (adj.sum()//2))

# path_1 = "examples/graph/OtherDataset/meta_modified_adj/wisconsin_normal.pt"
path_2 = "examples/graph/OtherDataset/meta_modified_adj/wisconsin_ptb_0.05.pt"
# original_adj = torch.load(path_1)
modified_adj = torch.load(path_2)
modified_adj = modified_adj.to(device)

def test(adj):

    edge_index = (adj > 0).nonzero().t()
    row, col = edge_index
    edge_weight = adj[row, col]
    # edge_weight = None  #这样引入了随机化，结果是随机的，有点奇怪

    gat = GAT(nfeat=features.shape[1],
              nhid=8, heads=8,
              nclass=labels.max().item() + 1,
              dropout=0.5, device=device)

    gat = gat.to(device)

    gat.fit(features, edge_index, edge_weight, labels, idx_train, idx_val, lip_norm=False, lip_node=False, lip_relu=False, gamma=0.008, verbose=True)  # train with earlystopping
    gat.test(features, edge_index, edge_weight, labels, idx_test)

def main():

    # print('=== testing GAT on original(clean) graph ===')
    # test(adj)
    print('=== testing GAT on modified(noisy) graph ===')
    test(modified_adj)

if __name__ == '__main__':
    main()




