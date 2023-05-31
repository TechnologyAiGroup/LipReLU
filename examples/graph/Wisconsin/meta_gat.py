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
parser.add_argument('--ptb_rate', type=float, default=0.04,  help='pertubation rate')      # [10 0.04 0.10 0.15 0.23 0.26]

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

modified_adj = attack_opology(features, adj, labels)       # 在没有扰动时，先给注销掉，避免随机种子的影响，使得结果可以复现
#
# path_1 = "examples/graph/OtherDataset/meta_modified_adj/wisconsin_normal.pt"
path_2 = "examples/graph/OtherDataset/meta_modified_adj/wisconsin_ptb_0.05.pt"

# torch.save(adj, path_1)
torch.save(modified_adj, path_2)

