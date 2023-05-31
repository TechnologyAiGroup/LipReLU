import torch
import argparse
from deeprobust.graph.defense.gcn_lip import GCN
from deeprobust.graph.global_attack import DICE
from deeprobust.graph.utils import *
from deeprobust.graph.global_attack import MinMax
import random
from scipy import sparse
from examples.graph.Wisconsin.loader import load_new_data
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
adj, features, labels = preprocess(adj, features, labels, preprocess_adj=False)
idx_train, idx_val, idx_test = get_data_split()

perturbations = int(args.ptb_rate * (adj.sum()//2))

def attack_opology(features, adj, labels):

    # Setup Victim Model

    victim_model = GCN(nfeat=features.shape[1], nclass=labels.max().item() + 1, nhid=16,
                       dropout=0.5, weight_decay=5e-4, device=device)

    victim_model = victim_model.to(device)
    victim_model.fit(features, adj, labels, idx_train, idx_val)
    # Setup Attack Model
    model = MinMax(model=victim_model, nnodes=adj.shape[0], loss_type='CE', device=device)
    model = model.to(device)
    model.attack(features, adj, labels, idx_train, perturbations)
    modified_adj = model.modified_adj
    return modified_adj

modified_adj = attack_opology(features, adj, labels)

# path_1 = "examples/graph/OtherDataset/topology_modified_adj/wisconsin_gat_normal.pt"
path_2 = "examples/graph/OtherDataset/topology_modified_adj/wisconsin_ptb_gat_0.25.pt"

# torch.save(adj, path_1)
torch.save(modified_adj, path_2)