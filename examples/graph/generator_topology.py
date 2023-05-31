from deeprobust.graph.data import Dataset
from deeprobust.graph.defense import GCN
from deeprobust.graph.global_attack import PGDAttack
from deeprobust.graph.utils import preprocess


data = Dataset(root='/tmp/', name='cora')
adj, features, labels = data.adj, data.features, data.labels
adj, features, labels = preprocess(adj, features, labels, preprocess_adj=False)  # conver to tensor
idx_train, idx_val, idx_test = data.idx_train, data.idx_val, data.idx_test
 # Setup Victim Model
victim_model = GCN(nfeat=features.shape[1], nclass=labels.max().item() + 1,
                        nhid=16, dropout=0.5, weight_decay=5e-4, device='cpu').to('cpu')
victim_model.fit(features, adj, labels, idx_train)
# Setup Attack Model
model = PGDAttack(model=victim_model, nnodes=adj.shape[0], loss_type='CE', device='cpu').to('cpu')
model.attack(features, adj, labels, idx_train, n_perturbations=10)
modified_adj = model.modified_adj
