import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import numpy as np
import pandas as panda
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


def dot_product_decode(Z1,Z2):
    A_pred = torch.sigmoid(torch.matmul(Z1,Z2.t()))
    return A_pred




def distance_decode(Z1,Z2):
    A_pred = torch.exp(-0.5*torch.cdist(Z1,Z2)**2)
    return A_pred

def glorot_init(input_dim, output_dim):
    init_range = np.sqrt(6.0/(input_dim + output_dim))
    initial = torch.rand(input_dim, output_dim)*2*init_range - init_range
    return nn.Parameter(initial)





class VBGAE(nn.Module):
    def __init__(self, adj,species):
        super(VBGAE,self).__init__()
        self.base_gcn1 = GraphConvSparse(args.input_dim1, args.hidden1_dim1, adj.T)
        self.gcn_mean1 = GraphConvSparse(args.hidden1_dim1, args.hidden2_dim1, adj, activation=lambda x:x)
        self.gcn_logstddev1 = GraphConvSparse(args.hidden1_dim1, args.hidden2_dim1, adj, activation=lambda x:x)
        
        self.combine1 = GraphConvSparse(args.hidden2_dim1,args.hidden2_dim1,species.T,activation=lambda x:x)

        self.base_gcn2 = GraphConvSparse(args.input_dim2, args.hidden1_dim2, adj)
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
    
    def forward(self,X1,X2):
        Z1 = self.encode1(X1)
        Z2 = self.encode2(X2)
        Z3 = self.combine1(Z1)
        A_pred2= dot_product_decode(Z3, Z2)
        A_pred= dot_product_decode(Z1, Z2)

        #A_pred2 = distance_decode(Z3, Z2)
        #A_pred = distance_decode(Z1, Z2)

        return A_pred,A_pred2, Z1, Z2,Z3
        
    
    
    

class VBGAE2(nn.Module):
    def __init__(self, adj,species):
        super(VBGAE2,self).__init__()
        self.base_gcn1 = GraphConvSparse(args.input_dim1, args.hidden1_dim1, adj.T)
        self.gcn_mean1 = GraphConvSparse(args.hidden1_dim1, args.hidden2_dim1, adj, activation=lambda x:x)
        self.gcn_logstddev1 = GraphConvSparse(args.hidden1_dim1, args.hidden2_dim1, adj, activation=lambda x:x)
        
        self.species1 = panda.DataFrame({0:np.arange(len(species)),1:species})

        self.base_gcn2 = GraphConvSparse(args.input_dim2, args.hidden1_dim2, adj)
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
    
    def encode3(self,X1):
        Z1 = self.encode1(X1)
        Z3 = Z1[self.species1.groupby(1).apply(lambda x: x.sample(1)).reset_index(drop=True)[0],:]
        return Z3
    
    def forward(self,X1,X2):
        Z1 = self.encode1(X1)
        Z2 = self.encode2(X2)
        Z3 = self.encode3(X1)
        #A_pred2= dot_product_decode(Z3, Z2)
        #A_pred= dot_product_decode(Z1, Z2)

        A_pred2 = distance_decode(Z3, Z2)
        A_pred = distance_decode(Z1, Z2)
        
        #A_pred2 = GRDPG_decode(Z3, Z2,2)
        #A_pred = GRDPG_decode(Z1, Z2,2)

        return A_pred,A_pred2, Z1, Z2,Z3
        
    

    
        
    
def init_parameters(net):
        for m in net.modules():
            if isinstance(m, nn.Linear):
                # nn.init.xavier_uniform(m.weight.data)
                torch.nn.init.normal_(m.weight.data)
                nn.init.constant_(m.bias.data, 0)
                
                
def GRDPG_decode(Z1,Z2,q=0):
    I_pq = torch.eye(Z1.shape[1])
    if q>0:
        I_pq[:,-q:] = -I_pq[:,-q:]
    A_pred = torch.sigmoid((Z1)@I_pq@Z2.T)
    return A_pred
       
    

class VBGAE3(nn.Module):
    def __init__(self, adj,species,GRDPG):
        super(VBGAE3,self).__init__()
        self.base_gcn1 = GraphConvSparse(args.input_dim1, args.hidden1_dim1, adj.T)
        self.gcn_mean1 = GraphConvSparse(args.hidden1_dim1, args.hidden2_dim1, adj, activation=lambda x:x)
        self.gcn_logstddev1 = GraphConvSparse(args.hidden1_dim1, args.hidden2_dim1, adj, activation=lambda x:x)
        
        self.species1 = panda.DataFrame({0:np.arange(len(species)),1:species})

        self.base_gcn2 = GraphConvSparse(args.input_dim2, args.hidden1_dim2, adj)
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
    
    def encode3(self,X1):
        Z1 = self.encode1(X1)
        Z3 = Z1[self.species1.groupby(1).apply(lambda x: x.sample(1)).reset_index(drop=True)[0],:]
        return Z3
    
    def forward(self,X1,X2):
        Z1 = self.encode1(X1)
        Z2 = self.encode2(X2)
        Z3 = self.encode3(X1)
    
        A_pred2 = GRDPG_decode(Z3, Z2,self.GRDPG)
        A_pred = GRDPG_decode(Z1, Z2,self.GRDPG)

        return A_pred,A_pred2, Z1, Z2,Z3
        
    
    
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
