import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import numpy as np

import args



class GraphConvSparse(nn.Module):
    def __init__(self, input_dim, output_dim, adj, activation = F.relu, **kwargs):
        super(GraphConvSparse, self).__init__(**kwargs)
        self.weight = glorot_init(input_dim, output_dim) 
        self.adj = adj
        self.activation = activation

    def forward(self, inputs):
        x = inputs
        x = torch.mm(x,self.weight)
        x = torch.mm(self.adj, x)
        outputs = self.activation(x)
        return outputs

class GraphConvSparse_adj(nn.Module):
    def __init__(self, input_dim, output_dim, activation = F.relu, **kwargs):
        super(GraphConvSparse_adj, self).__init__(**kwargs)
        self.weight = glorot_init(input_dim, output_dim) 
        self.activation = activation

    def forward(self, inputs,adj):
        x = inputs
        x = torch.mm(x,self.weight)
        x = torch.mm(adj, x)
        outputs = self.activation(x)
        return outputs



def dot_product_decode(Z1,Z2):
    A_pred = torch.sigmoid(torch.matmul(Z1,Z2.t()))
    return A_pred
def GRDPG_decode(Z1,Z2,q=0):
    I_pq = torch.eye(Z1.shape[1])
    if q>0:
        I_pq[:,-q:] = -I_pq[:,-q:]
    A_pred = torch.sigmoid((Z1)@I_pq@Z2.T)
    return A_pred
    
    

def distance_decode(Z1,Z2,scale=1):
    A_pred = torch.exp(-scale*0.5*torch.cdist(Z1,Z2)**2)
    return A_pred

def glorot_init(input_dim, output_dim):
    init_range = np.sqrt(6.0/(input_dim + output_dim))
    initial = torch.rand(input_dim, output_dim)*2*init_range - init_range
    return nn.Parameter(initial)



class VBGAE(nn.Module):
    def __init__(self, adj,input_dim1,input_dim2):
        super(VBGAE,self).__init__()
        self.base_gcn1 = GraphConvSparse(input_dim1, args.hidden1_dim1, adj.T)
        self.gcn_mean1 = GraphConvSparse(args.hidden1_dim1, args.hidden2_dim1, adj, activation=lambda x:x)
        self.gcn_logstddev1 = GraphConvSparse(args.hidden1_dim1, args.hidden2_dim1, adj, activation=lambda x:x)

        self.base_gcn2 = GraphConvSparse(input_dim2, args.hidden1_dim2, adj)
        self.gcn_mean2 = GraphConvSparse(args.hidden1_dim2, args.hidden2_dim2, adj.T, activation=lambda x:x)
        self.gcn_logstddev2 = GraphConvSparse(args.hidden1_dim2, args.hidden2_dim2, adj.T, activation=lambda x:x)

    def encode1(self, X1):
        hidden1 = self.base_gcn1(X1)
        self.mean1 = self.gcn_mean1(hidden1)
        self.logstd1 = self.gcn_logstddev1(hidden1)
        gaussian_noise1 = torch.randn(X1.size(0), args.hidden2_dim1)
        sampled_z1 = gaussian_noise1*torch.exp(self.logstd1) + self.mean1
        return sampled_z1

    def encode2(self, X2):
        hidden2 = self.base_gcn2(X2)
        self.mean2 = self.gcn_mean2(hidden2)
        self.logstd2 = self.gcn_logstddev2(hidden2)
        gaussian_noise2 = torch.randn(X2.size(0), args.hidden2_dim2)
        sampled_z2 = gaussian_noise2*torch.exp(self.logstd2) + self.mean2
        return sampled_z2
    
    def forward(self,X1,X2,scale=1):
        Z1 = self.encode1(X1)
        Z2 = self.encode2(X2)
        #A_pred= dot_product_decode(Z1, Z2)
        A_pred = distance_decode(Z1, Z2,scale)
        return A_pred, Z1, Z2

class VBGAE_GRDPG(nn.Module):
    def __init__(self, adj,input_dim1,input_dim2,GRDPG=0):
        super(VBGAE_GRDPG,self).__init__()
        self.base_gcn1 = GraphConvSparse(input_dim1, args.hidden1_dim1, adj.T)
        self.gcn_mean1 = GraphConvSparse(args.hidden1_dim1, args.hidden2_dim1, adj, activation=lambda x:x)
        self.gcn_logstddev1 = GraphConvSparse(args.hidden1_dim1, args.hidden2_dim1, adj, activation=lambda x:x)

        self.base_gcn2 = GraphConvSparse(input_dim2, args.hidden1_dim2, adj)
        self.gcn_mean2 = GraphConvSparse(args.hidden1_dim2, args.hidden2_dim2, adj.T, activation=lambda x:x)
        self.gcn_logstddev2 = GraphConvSparse(args.hidden1_dim2, args.hidden2_dim2, adj.T, activation=lambda x:x)
        self.GRDPG = GRDPG
    def encode1(self, X1):
        hidden1 = self.base_gcn1(X1)
        self.mean1 = self.gcn_mean1(hidden1)
        self.logstd1 = self.gcn_logstddev1(hidden1)
        gaussian_noise1 = torch.randn(X1.size(0), args.hidden2_dim1)
        sampled_z1 = gaussian_noise1*torch.exp(self.logstd1) + self.mean1
        return sampled_z1

    def encode2(self, X2):
        hidden2 = self.base_gcn2(X2)
        self.mean2 = self.gcn_mean2(hidden2)
        self.logstd2 = self.gcn_logstddev2(hidden2)
        gaussian_noise2 = torch.randn(X2.size(0), args.hidden2_dim2)
        sampled_z2 = gaussian_noise2*torch.exp(self.logstd2) + self.mean2
        return sampled_z2
    
    def forward(self,X1,X2,scale=1):
        Z1 = self.encode1(X1)
        Z2 = self.encode2(X2)
        A_pred = GRDPG_decode(Z1,Z2,self.GRDPG)
        return A_pred, Z1, Z2
        
class VBGAE_adj(nn.Module):
    def __init__(self,input_dim1,input_dim2):
        super(VBGAE_adj,self).__init__()
        self.base_gcn1 = GraphConvSparse_adj(input_dim1, args.hidden1_dim1)
        self.gcn_mean1 = GraphConvSparse_adj(args.hidden1_dim1, args.hidden2_dim1, activation=lambda x:x)
        self.gcn_logstddev1 = GraphConvSparse_adj(args.hidden1_dim1, args.hidden2_dim1, activation=lambda x:x)

        self.base_gcn2 = GraphConvSparse_adj(input_dim2, args.hidden1_dim2)
        self.gcn_mean2 = GraphConvSparse_adj(args.hidden1_dim2, args.hidden2_dim2, activation=lambda x:x)
        self.gcn_logstddev2 = GraphConvSparse_adj(args.hidden1_dim2, args.hidden2_dim2, activation=lambda x:x)

    def encode1(self, X1,adj):
        hidden1 = self.base_gcn1(X1,adj.T)
        self.mean1 = self.gcn_mean1(hidden1,adj)
        self.logstd1 = self.gcn_logstddev1(hidden1,adj)
        gaussian_noise1 = torch.randn(X1.size(0), args.hidden2_dim1)
        sampled_z1 = gaussian_noise1*torch.exp(self.logstd1) + self.mean1
        return sampled_z1

    def encode2(self, X2,adj):
        hidden2 = self.base_gcn2(X2,adj)
        self.mean2 = self.gcn_mean2(hidden2,adj.T)
        self.logstd2 = self.gcn_logstddev2(hidden2,adj.T)
        gaussian_noise2 = torch.randn(X2.size(0), args.hidden2_dim2)
        sampled_z2 = gaussian_noise2*torch.exp(self.logstd2) + self.mean2
        return sampled_z2
    
    def forward(self,X1,X2,adj):
        Z1 = self.encode1(X1,adj)
        Z2 = self.encode2(X2,adj)
        #A_pred= dot_product_decode(Z1, Z2)
        #A_pred = distance_decode(Z1, Z2)
        A_pred = GRDPG_decode(Z1, Z2,1)
        return A_pred, Z1, Z2
        
def projection_on_orthogonal(S,U):
    n= S.shape[0]
    inv = torch.inverse((S.T).matmul(S))
    proj = (torch.eye(n) - S.matmul(inv).matmul(S.T)).matmul(U)
    return proj

class VBGAE_proj(nn.Module):
    def __init__(self, adj,input_dim1,input_dim2):
        super(VBGAE_proj,self).__init__()
        self.base_gcn1 = GraphConvSparse(input_dim1, args.hidden1_dim1, adj.T)
        self.gcn_mean1 = GraphConvSparse(args.hidden1_dim1, args.hidden2_dim1, adj, activation=lambda x:x)
        self.gcn_logstddev1 = GraphConvSparse(args.hidden1_dim1, args.hidden2_dim1, adj, activation=lambda x:x)

        self.base_gcn2 = GraphConvSparse(input_dim2, args.hidden1_dim2, adj)
        self.gcn_mean2 = GraphConvSparse(args.hidden1_dim2, args.hidden2_dim2, adj.T, activation=lambda x:x)
        self.gcn_logstddev2 = GraphConvSparse(args.hidden1_dim2, args.hidden2_dim2, adj.T, activation=lambda x:x)

    def encode1(self, X1):
        hidden1 = self.base_gcn1(X1)
        self.mean1 = self.gcn_mean1(hidden1)
        self.logstd1 = self.gcn_logstddev1(hidden1)
        gaussian_noise1 = torch.randn(X1.size(0), args.hidden2_dim1)
        sampled_z1 = gaussian_noise1*torch.exp(self.logstd1) + self.mean1
        return sampled_z1

    def encode2(self, X2):
        hidden2 = self.base_gcn2(X2)
        self.mean2 = self.gcn_mean2(hidden2)
        self.logstd2 = self.gcn_logstddev2(hidden2)
        gaussian_noise2 = torch.randn(X2.size(0), args.hidden2_dim2)
        sampled_z2 = gaussian_noise2*torch.exp(self.logstd2) + self.mean2
        return sampled_z2
    
    def forward(self,X1,X2,S1,S2,scale=1):
        Z1 = projection_on_orthogonal(S1,self.encode1(X1))
        Z2 = projection_on_orthogonal(S2,self.encode2(X2))
        #A_pred= dot_product_decode(Z1, Z2)
        A_pred = distance_decode(Z1, Z2,scale)
        return A_pred, Z1, Z2    
    
    
    
    
def init_parameters(net):
        for m in net.modules():
            if isinstance(m, nn.Linear):
                # nn.init.xavier_uniform(m.weight.data)
                torch.nn.init.normal_(m.weight.data)
                nn.init.constant_(m.bias.data, 0)
                
             
class Adversary(nn.Module):
    def __init__(self,n_sensitive,n_hidden=32):
        super(Adversary, self).__init__()
        self.dense1 = nn.Linear(args.hidden2_dim1,n_hidden,bias=True)
        self.dense2 = nn.Linear(n_hidden,n_hidden,bias=True)
        self.dense3 = nn.Linear(n_hidden,n_hidden,bias=True)
        self.dense4 = nn.Linear(n_hidden,n_sensitive,bias=True)
        
    def forward(self,Z):
        s_hat = F.relu(self.dense1(Z))
        s_hat = F.relu(self.dense2(s_hat))
        s_hat = F.relu(self.dense3(s_hat))
        s_hat = self.dense4(s_hat)
        return(s_hat)


import torch
import torch.nn as nn
import torch.nn.functional as F

class GraphAttentionLayer(nn.Module):
    """
    Simple GAT layer, similar to https://arxiv.org/abs/1710.10903
    """
    def __init__(self, in_features, out_features, alpha, concat=True):
        super(GraphAttentionLayer, self).__init__()
        #self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat

        self.W = nn.Parameter(torch.empty(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.empty(size=(2*out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, h, adj):
        Wh = torch.mm(h, self.W) # h.shape: (N, in_features), Wh.shape: (N, out_features)
        e = self._prepare_attentional_mechanism_input(Wh)

        zero_vec = -9e15*torch.ones_like(adj)
        attention = torch.where(adj > 0, e, zero_vec)
        attention = F.softmax(attention, dim=1)
        #attention = F.dropout(attention, self.dropout, training=self.training)
        h_prime = torch.matmul(attention, Wh)

        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime

    def _prepare_attentional_mechanism_input(self, Wh):
        # Wh.shape (N, out_feature)
        # self.a.shape (2 * out_feature, 1)
        # Wh1&2.shape (N, 1)
        # e.shape (N, N)
        Wh1 = torch.matmul(Wh, self.a[:self.out_features, :])
        Wh2 = torch.matmul(Wh, self.a[self.out_features:, :])
        # broadcast add
        e = Wh1 + Wh2.T
        return self.leakyrelu(e)

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'

class GAT(nn.Module):
    def __init__(self, nfeat, nhid, nclass, alpha, nheads):
        """Dense version of GAT."""
        super(GAT, self).__init__()
        
        self.attentions = [GraphAttentionLayer(nfeat, nhid, alpha=alpha, concat=True) for _ in range(nheads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)

        self.out_att = GraphAttentionLayer(nhid * nheads, nclass, alpha=alpha, concat=False)

    def forward(self, x, adj):
        #x = F.dropout(x, self.dropout, training=self.training)
        x = torch.cat([att(x, adj) for att in self.attentions], dim=1)
        #x = F.dropout(x, self.dropout, training=self.training)
        x = self.out_att(x, adj)
        return x
      
        
      
        

class BGraphAttentionLayer(nn.Module):
    """
    Simple GAT layer, similar to https://arxiv.org/abs/1710.10903
    """
    def __init__(self, in_features1, in_features2, out_features,alpha, concat=True):
        super(BGraphAttentionLayer, self).__init__()
        #self.dropout = dropout
        self.in_features1 = in_features1
        self.in_features2 = in_features2
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat

        self.W1 = nn.Parameter(torch.empty(size=(in_features1, out_features)))
        nn.init.xavier_uniform_(self.W1.data, gain=1.414)
        
        self.W2 = nn.Parameter(torch.empty(size=(in_features2, out_features)))
        nn.init.xavier_uniform_(self.W2.data, gain=1.414)
        self.a = nn.Parameter(torch.empty(size=(2*out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)
        

        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, h1,h2, adj):
        Wh1 = torch.mm(h1, self.W1) # h.shape: (N, in_features), Wh.shape: (N, out_features)
        Wh2 = torch.mm(h2, self.W2)
        e = self._prepare_attentional_mechanism_input(Wh1,Wh2)

        zero_vec = -9e15*torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)
        attention1 = F.softmax(attention, dim=1)
        attention2 = F.softmax(attention, dim=0)
        #attention = F.dropout(attention, self.dropout, training=self.training)
        h_prime2 = torch.matmul(attention1.T, Wh1)
        h_prime1 = torch.matmul(attention2, Wh2)
        
        if self.concat:
            return F.elu(h_prime1), F.elu(h_prime2)
        else:
            return h_prime1,h_prime2

    def _prepare_attentional_mechanism_input(self, Wh1,Wh2):
        # Wh.shape (N, out_feature)
        # self.a.shape (2 * out_feature, 1)
        # Wh1&2.shape (N, 1)
        # e.shape (N, N)
        Wh_1 = torch.matmul(Wh1, self.a[:self.out_features, :])
        Wh_2 = torch.matmul(Wh2, self.a[self.out_features:, :])
        # broadcast add
        e = Wh_1.repeat(1,Wh2.shape[0]) + Wh_2.repeat(1,Wh1.shape[0]).T
        return self.leakyrelu(e)

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features1) + "+" + str(self.in_features2) + ' -> ' + str(self.out_features) + ')'

      
        
      
class BGAT(nn.Module):
    def __init__(self, nfeat1,nfeat2, nhid, nclass, alpha, nheads):
        """Dense version of GAT."""
        super(BGAT, self).__init__()
        
        self.attentions = [BGraphAttentionLayer(nfeat1,nfeat2 ,nhid, alpha=alpha, concat=True) for _ in range(nheads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)

        self.out_att = BGraphAttentionLayer(nhid * nheads,nhid * nheads , nclass, alpha=alpha, concat=False)
        
         
      
    def forward(self, x1,x2, adj):
        #x = F.dropout(x, self.dropout, training=self.training)
        X = [att(x1,x2, adj) for att in self.attentions]
        #x = F.dropout(x, self.dropout, training=self.training)
        X1 = torch.cat([x[0] for x in X],dim=1)
        X2 = torch.cat([x[1] for x in X],dim=1)
        
        res = self.out_att(X1,X2, adj)

        return res
        