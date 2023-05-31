import torch
import argparse
from deeprobust.graph.defense.gcn_lip import GCN
from deeprobust.graph.global_attack import DICE
from deeprobust.graph.utils import *
from deeprobust.graph.global_attack import MinMax
import random
from scipy import sparse
from examples.graph.OtherDataset.loader import load_new_data
from deeprobust.graph.defense.gat_lip_topo_pyg import GAT

parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=18, help='Random seed.')
parser.add_argument('--dataset', type=str, default='wisconsin', choices=['chameleon', 'cormelon', 'film', 'squirrel', 'texas', 'wisconsin'], help='dataset')
parser.add_argument('--ptb_rate', type=float, default=0.27,  help='perturbation rate')

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
adj, features, labels = preprocess(adj, features, labels, preprocess_adj=False, device=device)
idx_train, idx_val, idx_test = get_data_split()

perturbations = int(args.ptb_rate * (adj.sum()//2))

# path_1 = "examples/graph/OtherDataset/topology_modified_adj/wisconsin_gat_normal.pt"
path_2 = "examples/graph/OtherDataset/topology_modified_adj/wisconsin_ptb_gat_0.25.pt"
# original_adj = torch.load(path_1)
modified_adj = torch.load(path_2)

def test(adj):

    edge_index = (adj > 0).nonzero().t()
    row, col = edge_index
    edge_weight = adj[row, col]
    # edge_weight = None

    gat = GAT(nfeat=features.shape[1],
              nhid=8, heads=8,
              nclass=labels.max().item() + 1,
              dropout=0.5, device=device)

    gat = gat.to(device)

    gat.fit(features, edge_index, edge_weight, labels, idx_train, idx_val, lip_norm=False, lip_node=False, lip_relu=False, gamma=0.01, verbose=True)  # train with earlystopping
    gat.test(features, edge_index, edge_weight, labels, idx_test)

def main():

    # print('test on unperturbated adj of GAT')
    # test(adj)

    print('test on topology of GAT')
    test(modified_adj)


if __name__ == '__main__':
    main()
