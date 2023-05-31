"""
Extended from https://github.com/rusty1s/pytorch_geometric/tree/master/benchmark/citation
"""
import torch.nn as nn
import torch.nn.functional as F
import math
import time
import torch
import torch.optim as optim
import numpy as np
from torch import autograd
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
from deeprobust.graph import utils
from copy import deepcopy
from deeprobust.graph.defense.gat_lip_layers import GraphAttentionLayer
import argparse
import random


class GAT(nn.Module):
    """ 2 Layer Graph Attention Network based on pytorch geometric.

    Parameters
    ----------
    nfeat : int
        size of input feature dimension
    nhid : int
        number of hidden units
    nclass : int
        size of output dimension
    heads: int
        number of attention heads
    output_heads: int
        number of attention output heads
    dropout : float
        dropout rate for GAT
    lr : float
        learning rate for GAT
    weight_decay : float
        weight decay coefficient (l2 normalization) for GCN.
        When `with_relu` is True, `weight_decay` will be set to 0.
    with_bias: bool
        whether to include bias term in GAT weights.
    device: str
        'cpu' or 'cuda'.

    Examples
    --------
	We can first load dataset and then train GAT.

    >>> from deeprobust.graph.data import Dataset
    >>> from deeprobust.graph.defense import GAT
    >>> data = Dataset(root='/tmp/', name='cora')
    >>> adj, features, labels = data.adj, data.features, data.labels
    >>> idx_train, idx_val, idx_test = data.idx_train, data.idx_val, data.idx_test
    >>> gat = GAT(nfeat=features.shape[1],
              nhid=8, heads=8,
              nclass=labels.max().item() + 1,
              dropout=0.5, device='cpu')
    >>> gat = gat.to('cpu')
    >>> pyg_data = Dpr2Pyg(data) # convert deeprobust dataset to pyg dataset
    >>> gat.fit(pyg_data, patience=100, verbose=True) # train with earlystopping
    """

    def __init__(self, nfeat, nhid, nclass, heads=8, output_heads=1, dropout=0.5, lr=0.01,
            weight_decay=5e-4, with_bias=True, device=None):

        super(GAT, self).__init__()

        assert device is not None, "Please specify 'device'!"
        self.device = device
        self.nclass = nclass
        self.nfeat = nfeat
        self.hidden_sizes = [nhid]
        self.alpha = 0.2

        self.attentions = [GraphAttentionLayer(nfeat, nhid, dropout=dropout, alpha=self.alpha, device=self.device, concat=True) for _ in range(heads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)

        self.out_attention = GraphAttentionLayer(
            nhid * heads,
            nclass,
            dropout=dropout,
            alpha = self.alpha,
            device = device,
            concat=False
            )

        self.dropout = dropout

        self.weight_decay = weight_decay
        self.lr = lr
        self.output = None
        self.best_model = None
        self.best_output = None
        self.heads = heads
        self.out_heads = output_heads
        self.layer_1_input = None
        self.layer_1_output = None
        self.layer_final_input = None
        self.attention_list = None


        # self.attentions = []
        # self.attention_1 = GraphAttentionLayer(nfeat, nhid, dropout=dropout, alpha=self.alpha, device=self.device, concat=True)
        # self.attention_2 = GraphAttentionLayer(nfeat, nhid, dropout=dropout, alpha=self.alpha, device=self.device, concat=True)
        # self.attention_3 = GraphAttentionLayer(nfeat, nhid, dropout=dropout, alpha=self.alpha, device=self.device, concat=True)
        # self.attention_4 = GraphAttentionLayer(nfeat, nhid, dropout=dropout, alpha=self.alpha, device=self.device, concat=True)
        # self.attention_5 = GraphAttentionLayer(nfeat, nhid, dropout=dropout, alpha=self.alpha, device=self.device, concat=True)
        # self.attention_6 = GraphAttentionLayer(nfeat, nhid, dropout=dropout, alpha=self.alpha, device=self.device, concat=True)
        # self.attention_7 = GraphAttentionLayer(nfeat, nhid, dropout=dropout, alpha=self.alpha, device=self.device, concat=True)
        # self.attention_8 = GraphAttentionLayer(nfeat, nhid, dropout=dropout, alpha=self.alpha, device=self.device, concat=True)
        # self.attentions.append(self.attention_1)
        # self.attentions.append(self.attention_2)
        # self.attentions.append(self.attention_3)
        # self.attentions.append(self.attention_4)
        # self.attentions.append(self.attention_5)
        # self.attentions.append(self.attention_6)
        # self.attentions.append(self.attention_7)
        # self.attentions.append(self.attention_8)

    def forward(self, x, adj):
        x = F.dropout(x, self.dropout, training=self.training)
        x_list = []
        self.layer_1_output = x
        self.attention_list = []
        for att in self.attentions:
            layer_out, attention = att(x, adj)
            layer_out = F.elu(layer_out)
            x_list.append(layer_out)
            self.attention_list.append(attention)
        x = torch.cat(x_list, dim=1)    # 将各个自注意力层的输入连接在一起
        x = F.dropout(x, self.dropout, training=self.training)
        self.layer_final_input = x
        x, attention = self.out_attention(x, adj)
        self.attention_list.append(attention)
        return F.log_softmax(x, dim=1)

    # def initialize(self):
    #     """Initialize parameters of GAT.
    #     """
    #     self.conv1.reset_parameters()
    #     self.conv2.reset_parameters()

    # def lip_reg(self, features, adj, idx_train):
    #     lip_mat = []
    #     input = features.detach().clone()
    #     # input = input.to_dense()
    #     input.requires_grad_(True)
    #     output = self.forward(input, adj)
    #     for i in range(output.shape[1]):
    #         v = torch.zeros_like(output)
    #         v[:, i] = 1
    #         gradients = autograd.grad(outputs=output, inputs=input, grad_outputs=v,
    #                                   create_graph=True, retain_graph=True, only_inputs=True)[0]
    #         gradients = gradients[idx_train]  # 只能选取输入数据集的梯度
    #         grad_norm = torch.norm(gradients, dim=1).unsqueeze(dim=1)
    #         lip_mat.append(grad_norm)
    #
    #     input.requires_grad_(False)
    #     lip_concat = torch.cat(lip_mat, dim=1)
    #     lip_con_norm = torch.norm(lip_concat, dim=1)
    #     lip_loss = torch.max(lip_con_norm)
    #     return lip_loss

    # def get_attention_matrix(self, att_turple):
    #
    #     attention_list = []
    #     index = att_turple[0]
    #     attention_index = (index[0], index[1])
    #     attention_value = att_turple[1]
    #     N = int(attention_index.max()) + 1
    #     attention = torch.zeros((N,N))
    #     attention.index_put_(attention_index, attention_value)
    #     return attention

    # def get_model_param(self):
    #
    #     a_list = []
    #     w_list = []
    #     att_list = []
    #     att_dst_1 = torch.squeeze(self.conv1.att_dst)
    #     lin_src_weight_1 = self.conv1.lin_src.weight.T
    #     for i in range(self.heads):
    #         att_element = torch.unsqueeze(att_dst_1[i,:], dim=1)
    #         a_list.append(att_element)
    #         w_list.append(lin_src_weight_1[:,i*8:(i+1)*8])
    #
    #     att_dst_2 = torch.squeeze(self.conv2.att_dst)
    #     att_dst_2 = torch.unsqueeze(att_dst_2, dim=1)
    #     lin_src_weight_2 = self.conv2.lin_src.weight.T
    #     a_list.append(att_dst_2)
    #     w_list.append(lin_src_weight_2)
    #
    #     index_1 = self.attention_1[0]
    #     att_index_1 = (index_1[0], index_1[1])
    #     att_value_1 = self.attention_1[1]
    #     N = int(index_1.max()) + 1
    #
    #     for i in range(self.heads):
    #         attention = torch.zeros((N,N)).to(self.device)
    #         attention.index_put_(att_index_1, att_value_1[:,i])
    #         att_list.append(attention)
    #
    #     index_2 = self.attention_2[0]
    #     att_index_2 = (index_2[0], index_2[1])
    #     att_value_2 = self.attention_2[1]
    #     att_value_2 = torch.squeeze(att_value_2, dim=1)
    #     N = int(index_2.max()) + 1
    #     attention = torch.zeros((N, N)).to(self.device)
    #     attention.index_put_(att_index_2, att_value_2)
    #     att_list.append(attention)
    #
    #     self.attention_1 = None
    #     self.attention_2 = None
    #
    #     return a_list, w_list, att_list

    # def get_model_param(self):
    #
    #     a_list = []
    #     w_list = []
    #     att_list = []
    #     att_dst_1 = torch.squeeze(self.conv1.att_dst)
    #     lin_src_weight_1 = self.conv1.lin_src.weight.T
    #     for i in range(self.heads):
    #         a_list.append(att_dst_1[i,:])
    #         w_list.append(lin_src_weight_1[:,i*8:(i+1)*8])
    #
    #     att_dst_2 = torch.squeeze(self.conv2.att_dst)
    #     lin_src_weight_2 = self.conv2.lin_src.weight.T
    #     a_list.append(att_dst_2)
    #     w_list.append(lin_src_weight_2)
    #
    #     index_1 = self.attention_1[0]
    #     att_index_1 = (index_1[0], index_1[1])
    #     att_value_1 = self.attention_1[1]
    #     N = int(index_1.max()) + 1
    #
    #     for i in range(self.heads):
    #         attention = torch.zeros((N,N)).to(self.device)
    #         attention.index_put_(att_index_1, att_value_1[:,i])
    #         att_list.append(attention)
    #
    #     index_2 = self.attention_2[0]
    #     att_index_2 = (index_2[0], index_2[1])
    #     att_value_2 = self.attention_2[1]
    #     att_value_2 = torch.squeeze(att_value_2, dim=1)
    #     N = int(index_2.max()) + 1
    #     attention = torch.zeros((N, N)).to(self.device)
    #     attention.index_put_(att_index_2, att_value_2)
    #     att_list.append(attention)
    #
    #     self.attention_1 = None
    #     self.attention_2 = None
    #     return a_list, w_list, att_list

    # def get_layer_input(self):
    #
    #     layer_input = []
    #     layer_input.append(self.layer_1_input)
    #     layer_input.append(self.layer_final_input)
    #     self.layer_1_input = None
    #     self.layer_final_input = None
    #
    #     return layer_input

    # def get_layer_mask(self, output):
    #     output_grad_exp = torch.exp(output).detach().clone()
    #     output_grad_ones = torch.ones_like(output)
    #     mask = torch.where(output>0, output_grad_ones, output_grad_exp)
    #     return mask
    #
    # def lipschitz_compute_layer(self, W, a, X , attention):
    #
    #     v = torch.mm(a.T, W.T)
    #     V_norm = torch.norm(v, p=2)
    #     S_1 = torch.zeros((attention.shape[0], 1)).to(self.device)
    #     for i in range(attention.shape[0]):
    #         S_1[i, 0] = attention[i, i]
    #
    #     S_2 = S_1.repeat(1, W.shape[1])
    #     W_norm = torch.norm(W, dim=0).unsqueeze(dim=0)
    #     P_1 = torch.mm(S_1, W_norm)  #
    #     P_2 = S_2 * torch.mm(X, W)
    #     S_sum = torch.mm(attention, X)
    #     P_3 = S_2 * torch.mm(S_sum, W)
    #     M_4 = (P_2 - P_3) * V_norm + P_1
    #
    #     lip_layer_con = torch.norm(M_4, dim=1).unsqueeze(dim=1)
    #
    #     return lip_layer_con

    # def lipschitz_compute_layer_relu(self, W, a, X , attention):
    #
    #     v = torch.mm(a.T, W.T)
    #     V_norm = torch.norm(v, p=2)
    #     S_1 = torch.zeros((attention.shape[0], 1)).to(self.device)
    #     for i in range(attention.shape[0]):
    #         S_1[i, 0] = attention[i, i]
    #
    #     S_2 = S_1.repeat(1, W.shape[1])
    #     W_norm = torch.norm(W, dim=0).unsqueeze(dim=0)
    #     P_1 = torch.mm(S_1, W_norm)  #
    #     P_2 = S_2 * torch.mm(X, W)
    #     S_sum = torch.mm(attention, X)
    #     P_3 = S_2 * torch.mm(S_sum, W)
    #     lip_mat = (P_2 - P_3) * V_norm + P_1
    #
    #     return lip_mat

    # def lip_node(self, a_list, w_list, att_list):
    #
    #     lip_layer = []
    #     param_len = len(a_list) - 1  # 考虑多头自注意力层
    #     layer_input = self.get_layer_input()
    #     for i in range(param_len):
    #         a = a_list[i]
    #         W = w_list[i]
    #         lip_constant = self.lipschitz_compute_layer(W, a, layer_input[0], att_list[i])
    #         lip_layer.append(lip_constant)
    #
    #     lip_multi_layer = torch.cat(lip_layer, dim=1)
    #     lip_multi_layer_norm = torch.norm(lip_multi_layer, dim=1).unsqueeze(dim=1)
    #     # lip_multi_layer_max = torch.max(lip_multi_layer, dim=1)[0]
    #     # lip_multi_layer_max = torch.unsqueeze(lip_multi_layer_max, dim=1)
    #
    #     a = a_list[param_len]
    #     W = w_list[param_len]
    #     lip_final_constant = self.lipschitz_compute_layer(W, a, layer_input[1], att_list[param_len])
    #     lip_model_con = lip_multi_layer_norm * lip_final_constant
    #
    #     return max(lip_model_con)

    # def lip_relu(self, a_list, w_list, att_list):
    #
    #     lip_layer = []
    #     param_len = len(a_list) - 1  # 考虑多头自注意力层
    #     layer_input = self.get_layer_input()
    #     for i in range(param_len):
    #         a = a_list[i]
    #         W = w_list[i]
    #         lip_mat = self.lipschitz_compute_layer_relu(W, a, layer_input[0], att_list[i])
    #         lip_layer.append(lip_mat)
    #
    #     lip_multi_layer = torch.cat(lip_layer, dim=1)
    #     mask = self.get_layer_mask(self.layer_1_output)
    #     lip_multi_layer = lip_multi_layer * mask
    #     lip_multi_layer_norm = torch.norm(lip_multi_layer, dim=1).unsqueeze(dim=1)
    #     # lip_multi_layer_max = torch.max(lip_multi_layer, dim=1)[0]
    #     # lip_multi_layer_max = torch.unsqueeze(lip_multi_layer_max, dim=1)
    #
    #     a = a_list[param_len]
    #     W = w_list[param_len]
    #     lip_final_constant = self.lipschitz_compute_layer(W, a, layer_input[1], att_list[param_len])
    #     lip_model_con = lip_multi_layer_norm * lip_final_constant
    #
    #     return max(lip_model_con)

    def fit(self, features, adj, labels, idx_train, idx_val, train_iters=300, initialize=True, verbose=False, patience=100, lip_norm=False, lip_node=False, lip_relu=False, gamma=0.00, **kwargs):
        """Train the GAT model, when idx_val is not None, pick the best model
        according to the validation loss.

        Parameters
        ----------
        pyg_data :
            pytorch geometric dataset object
        train_iters : int
            number of training epochs
        initialize : bool
            whether to initialize parameters before training
        verbose : bool
            whether to show verbose logs
        patience : int
            patience for early stopping, only valid when `idx_val` is given
        """

        # if initialize:
        #     self.initialize()
        if type(adj) is not torch.Tensor:
            features, adj, labels = utils.to_tensor(features, adj, labels, device=self.device)
        else:
            features = features.to(self.device)
            adj = adj.to(self.device)
            labels = labels.to(self.device)

        self._train_with_val(features, adj, labels, idx_train, idx_val, train_iters, lip_norm, lip_node, lip_relu, gamma, verbose)

    def train_with_early_stopping(self, features, adj, labels, idx_train, idx_val, train_iters, patience, lip_norm, lip_node, lip_relu, gamma,  verbose):
        """early stopping based on the validation loss
        """
        if verbose:
            print('=== training GAT model ===')
        optimizer = optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        # labels = self.data.y
        # train_mask, val_mask = self.data.train_mask, self.data.val_mask

        early_stopping = patience
        best_loss_val = 100

        for i in range(train_iters):
            start_time = time.time()
            self.train()
            optimizer.zero_grad()
            output = self.forward(features, adj)
            loss_train = F.nll_loss(output[idx_train], labels[idx_train])

            # if lip_norm == True:
            #     loss_lip = self.lip_reg(features, adj, idx_train)
            #     loss_train = loss_train + gamma * loss_lip
            #
            # if lip_node == True:
            #     a_list, w_list, att_list = self.get_model_param()
            #     loss_lip = self.lip_node(a_list, w_list, att_list)
            #     loss_train = loss_train + gamma * loss_lip
            #
            # if lip_relu == True:
            #     a_list, w_list, att_list = self.get_model_param()
            #     loss_lip = self.lip_relu(a_list, w_list, att_list)
            #     loss_train = loss_train + gamma * loss_lip


            loss_train.backward()
            optimizer.step()

            if verbose and i % 1 == 0:
                print('Epoch {}, training loss: {}'.format(i, loss_train.item()))

            self.eval()
            output = self.forward(features, adj)
            loss_val = F.nll_loss(output[idx_val], labels[idx_val])

            if best_loss_val > loss_val:
                best_loss_val = loss_val
                self.output = output
                weights = deepcopy(self.state_dict())
                patience = early_stopping
            else:
                patience -= 1
            if i > early_stopping and patience <= 0:
                break
            print("the time of one epoch:", time.time() - start_time)

        if verbose:
             print('=== early stopping at {0}, loss_val = {1} ==='.format(i, best_loss_val) )

        self.load_state_dict(weights)



    def _train_with_val(self, features, adj, labels, idx_train, idx_val, train_iters, lip_norm, lip_node, lip_relu, gamma, verbose):
        if verbose:
            print('=== training GAT model ===')
        if verbose:
            print('=== picking the best model according to the performance on validation ===')

        optimizer = optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        # labels = self.data.y
        # train_mask, val_mask = self.data.train_mask, self.data.val_mask

        best_loss_val = 100
        best_acc_val = 0

        for i in range(train_iters):
            start_time = time.time()
            self.train()
            optimizer.zero_grad()
            output = self.forward(features, adj)
            loss_train = F.nll_loss(output[idx_train], labels[idx_train])

            # if lip_norm == True:
            #     loss_lip = self.lip_reg(self.data.x, self.data.edge_index, train_mask)
            #     loss_train = loss_train + gamma * loss_lip
            #
            # if lip_node == True:
            #     a_list, w_list, att_list = self.get_model_param()
            #     loss_lip = self.lip_node(a_list, w_list, att_list)
            #     loss_train = loss_train + gamma * loss_lip
            #
            # if lip_relu == True:
            #     a_list, w_list, att_list = self.get_model_param()
            #     loss_lip = self.lip_relu(a_list, w_list, att_list)
            #     loss_train = loss_train + gamma * loss_lip

            loss_train.backward()
            optimizer.step()

            if verbose and i % 1 == 0:
                print('Epoch {}, training loss: {} time: {}'.format(i, loss_train.item(), time.time()-start_time))

            self.eval()
            output = self.forward(features, adj)
            loss_val = F.nll_loss(output[idx_val], labels[idx_val])
            acc_val = utils.accuracy(output[idx_val], labels[idx_val])

            if best_loss_val > loss_val:
                best_loss_val = loss_val
                self.output = output
                weights = deepcopy(self.state_dict())

            if acc_val > best_acc_val:
                best_acc_val = acc_val
                self.output = output
                weights = deepcopy(self.state_dict())

        self.load_state_dict(weights)

    def test(self, features, adj, labels, idx_test):
        """Evaluate GAT performance on test set.

        Parameters
        ----------
        idx_test :
            node testing indices
        """
        if type(adj) is not torch.Tensor:
            features, adj, labels = utils.to_tensor(features, adj, labels, device=self.device)
        else:
            features = features.to(self.device)
            adj = adj.to(self.device)
            labels = labels.to(self.device)

        self.eval()
        output = self.forward(features, adj)
        # output = self.output
        loss_test = F.nll_loss(output[idx_test], labels[idx_test])
        acc_test = utils.accuracy(output[idx_test], labels[idx_test])
        print("Test set results:",
              "loss= {:.4f}".format(loss_test.item()),
              "accuracy= {:.4f}".format(acc_test.item()))
        return acc_test.item()

    def predict(self):
        """
        Returns
        -------
        torch.FloatTensor
            output (log probabilities) of GAT
        """

        self.eval()
        return self.forward(self.data)



# if __name__ == "__main__":
#     from deeprobust.graph.data import Dataset, Dpr2Pyg
#     # from deeprobust.graph.defense import GAT
#     data = Dataset(root='/tmp/', name='cora')
#     adj, features, labels = data.adj, data.features, data.labels
#     idx_train, idx_val, idx_test = data.idx_train, data.idx_val, data.idx_test
#     gat = GAT(nfeat=features.shape[1],
#           nhid=8, heads=8,
#           nclass=labels.max().item() + 1,
#           dropout=0.5, device='cpu')
#     gat = gat.to('cpu')
#     pyg_data = Dpr2Pyg(data)
#     gat.fit(pyg_data, lip_norm=False, gamma=0.001, verbose=True) # train with earlystopping
#     gat.test()
#     print(gat.predict())

