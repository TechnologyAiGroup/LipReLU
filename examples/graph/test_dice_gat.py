import torch
import numpy as np
import torch.nn.functional as F
import torch.optim as optim
from deeprobust.graph.defense.gcn_lip import GCN
from deeprobust.graph.global_attack import DICE
from deeprobust.graph.data import Dataset, Dpr2Pyg
from deeprobust.graph.defense.gat_lip import GAT
from deeprobust.graph.utils import *
from deeprobust.graph.data import Dataset
import random
import scipy

import argparse

parser = argparse.ArgumentParser()
# parser.add_argument('--seed', type=int, default=15, help='Random seed.') # The seed of gcn
parser.add_argument('--seed', type=int, default=20, help='Random seed.') # The seed of gat
parser.add_argument('--dataset', type=str, default='cora', choices=['cora', 'cora_ml', 'citeseer', 'polblogs', 'pubmed'], help='dataset')
parser.add_argument('--ptb_rate', type=float, default=0.25,  help='pertubation rate')


args = parser.parse_args()
args.cuda = torch.cuda.is_available()
print('cuda: %s' % args.cuda)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# np.random.seed(args.seed)
# torch.manual_seed(args.seed)
# if args.cuda:
#     torch.cuda.manual_seed(args.seed)
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

# >>> from deeprobust.graph.data import Dataset
# >>> from deeprobust.graph.global_attack import DICE
# >>> data = Dataset(root='/tmp/', name='cora')
# >>> adj, features, labels = data.adj, data.features, data.labels
# >>> model = DICE()
# >>> model.attack(adj, labels, n_perturbations=10)
# >>> modified_adj = model.modified_adj

# # data = Dataset(root='/tmp/', name=args.dataset)
data = Dataset(root='/tmp/', name=args.dataset)
adj, features, labels = data.adj, data.features, data.labels
idx_train, idx_val, idx_test = data.idx_train, data.idx_val, data.idx_test
idx_unlabeled = np.union1d(idx_val, idx_test)
#
# Setup Attack Model
model = DICE()
n_perturbations = int(args.ptb_rate * (adj.sum()//2))
model.attack(adj, labels, n_perturbations)

path_1 = "examples/graph/dice_modified_adj/gat_cora_dice_normal.pt"
path_2 = "examples/graph/dice_modified_adj/citeseer_dice_attack_0.25.pt"

# modified_adj = model.modified_adj
# torch.save(adj, path_1)
# torch.save(modified_adj, path_2)
# modified_adj = torch.load(path_2)

#
def test_GAT(adj):


    gat = GAT(nfeat=features.shape[1],
              nhid=8, heads=8,
              nclass=labels.max().item() + 1,
              dropout=0.5, device=device)

    pyg_data = Dpr2Pyg(data)
    pyg_data.update_edge_index(adj)  # inplace operation
    gat = gat.to(device)
    gat.fit(pyg_data, train_iters=300, lip_norm=False, lip_node=False, lip_relu=False, gamma=0.0039, verbose=True) # train with earlystopping
    gat.test()

def main():

    # print('test on unperturbated adj of GAT')
    # perturbed_adj = adj
    # test_GAT(perturbed_adj)

    print('test on dice of GAT')
    perturbed_adj = model.modified_adj
    test_GAT(perturbed_adj)

if __name__ == '__main__':
    main()
#
