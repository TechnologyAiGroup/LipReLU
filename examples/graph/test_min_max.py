import torch
import numpy as np
import torch.nn.functional as F
import torch.optim as optim
# from deeprobust.graph.defense import GCN
from deeprobust.graph.defense.gcn_lip import GCN
from deeprobust.graph.global_attack import MinMax
from deeprobust.graph.utils import *
from deeprobust.graph.data import Dataset
from deeprobust.graph.utils import preprocess
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
parser.add_argument('--model', type=str, default='PGD', choices=['PGD', 'min-max'], help='model variant')

args = parser.parse_args()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# device = 'cpu'

np.random.seed(args.seed)
torch.manual_seed(args.seed)
if device != 'cpu':
    torch.cuda.manual_seed(args.seed)

data = Dataset(root='/tmp/', name=args.dataset)
adj, features, labels = data.adj, data.features, data.labels
adj, features, labels = preprocess(adj, features, labels, preprocess_adj=False)

idx_train, idx_val, idx_test = data.idx_train, data.idx_val, data.idx_test
# idx_unlabeled = np.union1d(idx_val, idx_test)

perturbations = int(args.ptb_rate * (adj.sum()//2))

# Setup Victim Model
victim_model = GCN(nfeat=features.shape[1], nclass=labels.max().item()+1, nhid=16,
        dropout=0.5, weight_decay=5e-4, device=device)

victim_model = victim_model.to(device)
victim_model.fit(features, adj, labels, idx_train)
#
# # Setup Attack Model
#
model = MinMax(model=victim_model, nnodes=adj.shape[0], loss_type='CE', device=device)
model = model.to(device)
model.attack(features, adj, labels, idx_train, perturbations)
modified_adj = model.modified_adj

def test(adj):
    ''' test on GCN '''

    # adj = normalize_adj_tensor(adj)
    gcn = GCN(nfeat=features.shape[1],
              nhid=args.hidden,
              nclass=labels.max().item() + 1,
              dropout=args.dropout, device=device)
    gcn = gcn.to(device)
    gcn.fit(features, adj, labels, idx_train, train_iters=200, lip_norm=False, gamma=0.1) # train without model picking
    # gcn.fit(features, adj, labels, idx_train, idx_val) # train with validation model picking
    output = gcn.output.to('cpu')
    loss_test = F.nll_loss(output[idx_test], labels[idx_test])
    acc_test = accuracy(output[idx_test], labels[idx_test])
    print("Test set results:",
          "loss= {:.4f}".format(loss_test.item()),
          "accuracy= {:.4f}".format(acc_test.item()))
    return acc_test.item()

def main():

    print('=== testing GCN on original(clean) graph ===')
    test(adj)

    # modified_features = model.modified_features
    print('=== testing GCN on modified(attack) graph ===')
    test(modified_adj)

    # # if you want to save the modified adj/features, uncomment the code below
    # model.save_adj(root='./', name=f'mod_adj')
    # model.save_features(root='./', name='mod_features')


if __name__ == '__main__':
    main()

