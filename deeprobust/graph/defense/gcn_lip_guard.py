import time

import torch.nn as nn
import torch.nn.functional as F
import math
import torch
import torch.optim as optim
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
from deeprobust.graph import utils
from copy import deepcopy
from torch import autograd
from torch_geometric.nn import GINConv, GATConv, GCNConv, JumpingKnowledge
import numpy as np
from sklearn.metrics.pairwise import euclidean_distances, cosine_similarity
from sklearn.metrics import f1_score
from scipy.sparse import lil_matrix
from scipy.sparse.construct import diags
from sklearn.preprocessing import normalize

class GraphConvolution(Module):
    """Simple GCN layer, similar to https://github.com/tkipf/pygcn
    """

    def __init__(self, in_features, out_features, with_bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if with_bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        """ Graph Convolutional Layer forward function
        """
        if input.data.is_sparse:
            support = torch.spmm(input, self.weight)
        else:
            support = torch.mm(input, self.weight)
        output = torch.spmm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


class GCN(nn.Module):
    """ 2 Layer Graph Convolutional Network.

    Parameters
    ----------
    nfeat : int
        size of input feature dimension
    nhid : int
        number of hidden units
    nclass : int
        size of output dimension
    dropout : float
        dropout rate for GCN
    lr : float
        learning rate for GCN
    weight_decay : float
        weight decay coefficient (l2 normalization) for GCN.
        When `with_relu` is True, `weight_decay` will be set to 0.
    with_relu : bool
        whether to use relu activation function. If False, GCN will be linearized.
    with_bias: bool
        whether to include bias term in GCN weights.
    device: str
        'cpu' or 'cuda'.

    Examples
    --------
	We can first load dataset and then train GCN.

    >>> from deeprobust.graph.data import Dataset
    >>> from deeprobust.graph.defense import GCN
    >>> data = Dataset(root='/tmp/', name='cora')
    >>> adj, features, labels = data.adj, data.features, data.labels
    >>> idx_train, idx_val, idx_test = data.idx_train, data.idx_val, data.idx_test
    >>> gcn = GCN(nfeat=features.shape[1],
              nhid=16,
              nclass=labels.max().item() + 1,
              dropout=0.5, device='cpu')
    >>> gcn = gcn.to('cpu')
    >>> gcn.fit(features, adj, labels, idx_train) # train without earlystopping
    >>> gcn.fit(features, adj, labels, idx_train, idx_val, patience=30) # train with earlystopping
    >>> gcn.test(idx_test)
    """

    def __init__(self, nfeat, nhid, nclass, dropout=0.5, lr=0.01, weight_decay=5e-4,
                 with_relu=True, with_bias=True, device=None):

        super(GCN, self).__init__()

        assert device is not None, "Please specify 'device'!"
        self.device = device
        self.nfeat = nfeat
        self.hidden_sizes = [nhid]
        self.nclass = nclass
        self.gc1 = GraphConvolution(nfeat, nhid, with_bias=with_bias)
        self.gc2 = GraphConvolution(nhid, nclass, with_bias=with_bias)

        self.dropout = dropout
        self.lr = lr
        self.weight_decay  = weight_decay
        self.with_relu = with_relu
        self.with_bias = with_bias
        self.output = None
        self.best_model = None
        self.best_output = None
        self.adj_norm = None
        self.features = None
        self.gc1_relu = None
        self.gc2_out = None

    def forward(self, x, adj):
        if self.with_relu:
            x = F.relu(self.gc1(x, adj))
            self.gc1_relu = x
        else:
            x = self.gc1(x, adj)

        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, adj)
        self.gc2_out = x
        return F.log_softmax(x, dim=1)


    def initialize(self):
        """Initialize parameters of GCN.
        """
        self.gc1.reset_parameters()
        self.gc2.reset_parameters()

    #########################
    def get_model_param(self):

        W = []
        W.append(self.gc1.weight)
        W.append(self.gc2.weight)
        return W, len(W)

    def sup_lip_constant(self, weight, adj):

        weight_norm = torch.norm(weight, dim=0).unsqueeze(dim=0)

        lip_mat = torch.mm(adj, weight_norm)

        # lip_mat_norm = torch.norm(lip_mat,dim=1).unsqueeze(dim=1)

        return lip_mat

    def lip_regulation_node(self, features, adj):  # 计算lipschitz，非梯度，即网络的lipschitz常数
        # print(adj)
        # print(type(adj))
        adj_diag = torch.zeros((adj.shape[0], 1)).to(adj.device)  # 用来记录Adj对角元素的值

        for i in range(adj.shape[0]):
            adj_diag[i, 0] = adj[i, i]

        lip_con = torch.ones((features.shape[0], 1)).to(adj.device)

        W_list, layer_len = self.get_model_param()

        for i in range(layer_len):
            W = W_list[i]
            lip_mat = self.sup_lip_constant(W, adj_diag)
            lip_mat_norm = torch.norm(lip_mat, dim=1).unsqueeze(dim=1)
            lip_con = lip_con * lip_mat_norm

        lip_constant = torch.max(lip_con)  # 这处不要改成 max(lip_con)
        print("lip_constant:", lip_constant)
        return lip_constant

    ##########################
    def get_out_mask(self, output):

        mask = torch.where(output > 0.0, 1.0, 0.0)
        return mask

    def lip_regulation_relu(self, features, adj):  # 添加激活函数relu,非梯度

        layer_output = []
        layer_output.append(self.gc1_relu)
        layer_output.append(self.gc2_out)

        adj_diag = torch.zeros((adj.shape[0], 1)).to(adj.device)  # 用来记录Adj对角元素的值

        for i in range(adj.shape[0]):
            adj_diag[i, 0] = adj[i, i]

        lip_con = torch.ones((features.shape[0], 1)).to(adj.device)

        W_list, layer_len = self.get_model_param()

        layer_next_relu = [0]
        for i in range(layer_len):
            W = W_list[i]
            if i in layer_next_relu:
                mask = self.get_out_mask(layer_output[i])
            else:
                mask = torch.ones_like(layer_output[i])
            lip_mat = self.sup_lip_constant(W, adj_diag) * mask
            lip_mat_norm = torch.norm(lip_mat, dim=1).unsqueeze(dim=1)
            lip_con = lip_con * lip_mat_norm

        lip_constant = torch.max(lip_con)  # 这处不要改成 max(lip_con)
        print("lip_constant:", lip_constant)
        return lip_constant

    ##########################
    def lip_reg(self, features, adj, idx_train):  # Lipschitz_Grad 计算lipschitz常数

        lip_mat = []
        input = features.detach().clone()
        input = input.to_dense()  # 在test_MIN_MAX需要将其注销掉
        input.requires_grad_(True)
        output = self.forward(input, adj)
        for i in range(output.shape[1]):
            v = torch.zeros_like(output)
            v[:, i] = 1
            gradients = autograd.grad(outputs=output, inputs=input, grad_outputs=v,
                                      create_graph=True, retain_graph=True, only_inputs=True)[0]
            gradients = gradients[idx_train]  # 只能选取输入数据集的梯度
            grad_norm = torch.norm(gradients, dim=1).unsqueeze(dim=1)
            lip_mat.append(grad_norm)

        input.requires_grad_(False)
        lip_concat = torch.cat(lip_mat, dim=1)
        lip_con_norm = torch.norm(lip_concat, dim=1)
        lip_loss = torch.max(lip_con_norm)
        return lip_loss

    ###########################

    def fit(self, features, adj, labels, idx_train, idx_val=None, train_iters=200, initialize=True,
            verbose=False, normalize=True, patience=500, lip_norm=False, lip_node=False, lip_relu=False, gamma=0.0,
            **kwargs):
        """Train the gcn model, when idx_val is not None, pick the best model according to the validation loss.

        Parameters
        ----------
        features :
            node features
        adj :
            the adjacency matrix. The format could be torch.tensor or scipy matrix
        labels :
            node labels
        idx_train :
            node training indices
        idx_val :
            node validation indices. If not given (None), GCN training process will not adpot early stopping
        train_iters : int
            number of training epochs
        initialize : bool
            whether to initialize parameters before training
        verbose : bool
            whether to show verbose logs
        normalize : bool
            whether to normalize the input adjacency matrix.
        patience : int
            patience for early stopping, only valid when `idx_val` is given
        """
        self.sim = None
        self.attention = attention

        self.device = self.gc1.weight.device
        if initialize:
            self.initialize()

        if type(adj) is not torch.Tensor:
            features, adj, labels = utils.to_tensor(features, adj, labels, device=self.device)
        else:
            features = features.to(self.device)
            adj = adj.to(self.device)
            labels = labels.to(self.device)

        if normalize:
            if utils.is_sparse_tensor(adj):
                adj_norm = utils.normalize_adj_tensor(adj, sparse=True)
            else:
                adj_norm = utils.normalize_adj_tensor(adj)
        else:
            adj_norm = adj

        self.adj_norm = adj_norm
        self.features = features
        self.labels = labels

        self._train_with_val(labels, idx_train, idx_val, train_iters, lip_norm, lip_node, lip_relu, gamma, verbose)

        # if idx_val is None:
        #     self._train_without_val(labels, idx_train, train_iters, lip_norm, lip_node, lip_relu, gamma, verbose)
        # else:
        #     if patience < train_iters:
        #         self._train_with_early_stopping(labels, idx_train, idx_val, train_iters, patience, lip_norm, lip_node , lip_relu, gamma, verbose)
        #     else:
        #         self._train_with_val(labels, idx_train, idx_val, train_iters, lip_norm, lip_node, lip_relu, gamma, verbose)

    def _train_without_val(self, labels, idx_train, train_iters, lip_norm, lip_node, lip_relu, gamma, verbose):
        self.train()
        optimizer = optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        for i in range(train_iters):

            optimizer.zero_grad()
            output = self.forward(self.features, self.adj_norm)
            loss_train = F.nll_loss(output[idx_train], labels[idx_train])

            if lip_norm == True:
                loss_lip = self.lip_reg(self.features, self.adj_norm, idx_train)
                loss_train = loss_train + gamma * loss_lip

            if lip_node == True:
                loss_lip = self.lip_regulation_node(self.features, self.adj_norm)
                loss_train = loss_train + gamma * loss_lip

            if lip_relu == True:
                loss_lip = self.lip_regulation_relu(self.features, self.adj_norm)
                loss_train = loss_train + gamma * loss_lip

            loss_train.backward()
            optimizer.step()
            if verbose and i % 10 == 0:
                print('Epoch {}, training loss: {}'.format(i, loss_train.item()))

        self.eval()
        output = self.forward(self.features, self.adj_norm)
        self.output = output

    def _train_with_val(self, labels, idx_train, idx_val, train_iters, lip_norm, lip_node, lip_relu, gamma, verbose):

        if verbose:
            print('=== training gcn model ===')
        optimizer = optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        best_loss_val = 100
        best_acc_val = 0

        for i in range(train_iters):
            start_time = time.time()
            self.train()
            optimizer.zero_grad()
            output = self.forward(self.features, self.adj_norm)
            loss_train = F.nll_loss(output[idx_train], labels[idx_train])
            lip_constant = None
            if lip_norm == True:
                loss_lip = self.lip_reg(self.features, self.adj_norm, idx_train)
                lip_constant = loss_lip
                loss_train = loss_train + gamma * loss_lip

            if lip_node == True:
                loss_lip = self.lip_regulation_node(self.features, self.adj_norm)
                lip_constant = loss_lip
                loss_train = loss_train + gamma * loss_lip

            if lip_relu == True:
                loss_lip = self.lip_regulation_relu(self.features, self.adj_norm)
                lip_constant = loss_lip
                loss_train = loss_train + gamma * loss_lip
            # else:
            #     loss_lip = self.lip_regulation_relu(self.features, self.adj_norm)
            #     lip_constant = loss_lip

            loss_train.backward()
            optimizer.step()

            if verbose and i % 1 == 0:
                print('Epoch {}, training loss: {}, time: {}, lip_constant: {}'.format(i, loss_train.item(),
                                                                                       time.time() - start_time,
                                                                                       lip_constant))

            self.eval()
            output = self.forward(self.features, self.adj_norm)
            loss_val = F.nll_loss(output[idx_val], labels[idx_val])
            acc_val = utils.accuracy(output[idx_val], labels[idx_val])

            if best_loss_val > loss_val:
                best_loss_val = loss_val
                self.output = output
                weights = deepcopy(self.state_dict())
                best_lip = lip_constant

            if acc_val > best_acc_val:
                best_acc_val = acc_val
                self.output = output
                weights = deepcopy(self.state_dict())
                best_lip = lip_constant

        if verbose:
            print('model lipschitz constant: ', best_lip)
            print('=== picking the best model according to the performance on validation ===')
        self.load_state_dict(weights)

    def _train_with_early_stopping(self, labels, idx_train, idx_val, train_iters, patience, lip_norm, lip_node,
                                   lip_relu, gamma, verbose):
        if verbose:
            print('=== training gcn model ===')
        optimizer = optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        early_stopping = patience
        best_loss_val = 100

        for i in range(train_iters):
            self.train()
            optimizer.zero_grad()
            output = self.forward(self.features, self.adj_norm)
            loss_train = F.nll_loss(output[idx_train], labels[idx_train])
            if lip_norm == True:
                loss_lip = self.lip_reg(self.features, self.adj_norm, idx_train)
                loss_train = loss_train + gamma * loss_lip

            if lip_node == True:
                loss_lip = self.lip_regulation_node(self.features, self.adj_norm)
                loss_train = loss_train + gamma * loss_lip

            if lip_relu == True:
                loss_lip = self.lip_regulation_relu(self.features, self.adj_norm)
                loss_train = loss_train + gamma * loss_lip

            loss_train.backward()
            optimizer.step()

            if verbose and i % 10 == 0:
                print('Epoch {}, training loss: {}'.format(i, loss_train.item()))

            self.eval()
            output = self.forward(self.features, self.adj_norm)

            # def eval_class(output, labels):
            #     preds = output.max(1)[1].type_as(labels)
            #     return f1_score(labels.cpu().numpy(), preds.cpu().numpy(), average='micro') + \
            #         f1_score(labels.cpu().numpy(), preds.cpu().numpy(), average='macro')

            # perf_sum = eval_class(output[idx_val], labels[idx_val])
            loss_val = F.nll_loss(output[idx_val], labels[idx_val])

            if best_loss_val > loss_val:
                best_loss_val = loss_val
                self.output = output
                weights = deepcopy(self.state_dict())
                patience = early_stopping  # 每次在early_stopping规定的次数之外找到更好的模型，对其进行重置
            else:
                patience -= 1
            if i > early_stopping and patience <= 0:  # 如果训练次数i小于early_stopping，即使满足了条件，也要继续训练
                break

        if verbose:
            print('=== early stopping at {0}, loss_val = {1} ==='.format(i, best_loss_val))
        self.load_state_dict(weights)

    def test(self, idx_test):
        self.eval()
        output = self.predict()  # here use the self.features and self.adj_norm in training stage
        loss_test = F.nll_loss(output[idx_test], self.labels[idx_test])
        acc_test = utils.accuracy(output[idx_test], self.labels[idx_test])
        print("Test set results:",
              "loss= {:.4f}".format(loss_test.item()),
              "accuracy= {:.4f}".format(acc_test.item()))
        return acc_test, output

    def predict(self, features=None, adj=None):
        """By default, the inputs should be unnormalized adjacency

        Parameters
        ----------
        features :
            node features. If `features` and `adj` are not given, this function will use previous stored `features` and `adj` from training to make predictions.
        adj :
            adjcency matrix. If `features` and `adj` are not given, this function will use previous stored `features` and `adj` from training to make predictions.


        Returns
        -------
        torch.FloatTensor
            output (log probabilities) of GCN
        """

        self.eval()
        if features is None and adj is None:
            return self.forward(self.features, self.adj_norm)
        else:
            if type(adj) is not torch.Tensor:
                features, adj = utils.to_tensor(features, adj, device=self.device)

            self.features = features
            if utils.is_sparse_tensor(adj):
                self.adj_norm = utils.normalize_adj_tensor(adj, sparse=True)
            else:
                self.adj_norm = utils.normalize_adj_tensor(adj)
            return self.forward(self.features, self.adj_norm)











