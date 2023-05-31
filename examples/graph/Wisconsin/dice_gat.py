import torch
import argparse
from deeprobust.graph.defense.gcn_lip import GCN
from deeprobust.graph.global_attack import DICE
from deeprobust.graph.utils import *
import random
from scipy import sparse
from examples.graph.OtherDataset.loader import load_new_data
from deeprobust.graph.defense.gat_lip_topo_pyg import GAT
from deeprobust.graph.global_attack import DICE

parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=18, help='Random seed.')
parser.add_argument('--dataset', type=str, default='wisconsin', choices=['chameleon', 'cormelon', 'film', 'squirrel', 'texas', 'wisconsin'], help='dataset')
parser.add_argument('--ptb_rate', type=float, default=0.10,  help='perturbation rate')   # 0.01 0.11 0.15 0.20 0.25

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
idx_unlabeled = np.union1d(idx_val, idx_test)
perturbations = int(args.ptb_rate * (adj.sum()//2))

def dice_attack(adj, labels):

    model = DICE()
    n_perturbations = int(args.ptb_rate * (adj.sum() // 2))
    model.attack(adj, labels, n_perturbations)
    modified_adj = model.modified_adj
    modified_adj = sparse_mx_to_torch_sparse_tensor(modified_adj)
    modified_adj = modified_adj.to_dense()
    return modified_adj

modified_adj = dice_attack(adj, labels)
modified_adj = modified_adj.to(device)
adj, features, labels = preprocess(adj, features, labels, preprocess_adj=False)
adj = adj.to(device)
features = features.to(device)
labels = labels.to(device)

# path_1 = "examples/graph/gat_dice_modified_adj/gat_cora_dice_normal.pt"
# path_2 = "examples/graph/gat_dice_modified_adj/cora_dice_attack_0.10.pt"

# modified_adj = model.modified_adj
# torch.save(adj, path_1)
# torch.save(modified_adj, path_2)

#
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

    # print('test on unperturbated adj of GAT')
    # test(adj)

    print('test on dice of GAT')
    test(modified_adj)


if __name__ == '__main__':
    main()
# #
