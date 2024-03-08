'''
****************NOTE*****************
CREDITS : Thomas Kipf
since datasets are the same as those in kipf's implementation, 
Their preprocessing source was used as-is.
*************************************
'''
import numpy as np
import scipy.sparse as sp
from sklearn.metrics import roc_auc_score, average_precision_score
import torch

def sparse_to_tuple(sparse_mx):
    if not sp.isspmatrix_coo(sparse_mx):
        sparse_mx = sparse_mx.tocoo()
    coords = np.vstack((sparse_mx.row, sparse_mx.col)).transpose()
    values = sparse_mx.data
    shape = sparse_mx.shape
    return coords, values, shape

def preprocess_graph(adj):
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    colsum = np.array(adj.sum(0))
    D1 = sp.diags(np.power(rowsum+1, -0.5).flatten())
    D2 = sp.diags(np.power(colsum+1, -0.5).flatten())
    adj_normalized = D1.dot(adj).dot(D2).tocoo()
    return sparse_to_tuple(adj_normalized)


def mask_test_edges(adj):
    # Function to build test set with 10% positive links

    edges = np.where(adj>0)
    non_edges = np.where(adj==0)
    
    permut_edges= np.random.permutation(edges[0].shape[0])
    edges = edges[0][permut_edges],edges[1][permut_edges]
    
    permut_non_edges= np.random.permutation(non_edges[0].shape[0])
    non_edges =  non_edges[0][permut_non_edges],non_edges[1][permut_non_edges]
    
    num_test = int(np.floor(edges[0].shape[0] / 10.))
    num_val  = int(np.floor(edges[0].shape[0] / 20.))
    
    edges = np.split(edges[0],[num_test,num_test+num_val]),np.split(edges[1],[num_test,num_test+num_val])
    non_edges = np.split(non_edges[0],[num_test,num_test+num_val]),np.split(non_edges[1],[num_test,num_test+num_val])
    
    train_edges,val_edges,test_edges = (edges[0][2],edges[1][2]), (edges[0][1],edges[1][1]), (edges[0][0],edges[1][0])
    val_edges_false,test_edges_false = (non_edges[0][1],non_edges[1][1]),(non_edges[0][0],non_edges[1][0])
    
    data = np.ones(train_edges[0].shape[0])
    adj_train = sp.csr_matrix((data, train_edges), shape=adj.shape)

    return adj_train, train_edges, val_edges, val_edges_false, test_edges, test_edges_false




def get_scores(val_edges,val_edges_false, A_pred):

    pos_pred=A_pred[val_edges].detach().numpy()
    neg_pred=A_pred[val_edges_false].detach().numpy()

    preds_all = np.hstack([pos_pred, neg_pred])
    labels_all = np.hstack([np.ones(len(pos_pred)), np.zeros(len(neg_pred))])
    roc_score = roc_auc_score(labels_all, preds_all)
    ap_score = average_precision_score(labels_all, preds_all)

    return roc_score, ap_score



def get_acc(A_pred, adj_label):
    labels_all = adj_label.to_dense().view(-1).long()
    preds_all = (A_pred > 0.5).view(-1).long()
    accuracy = (preds_all == labels_all).sum().float() / labels_all.size(0)
    return accuracy

def mask_test_edges2(adj_label,species, val_edges, val_edges_false, test_edges, test_edges_false):
        
    bipartite = 1*(species.T.dot(adj_label.to_dense().numpy())>0)
    forbidden = (bipartite==1)
    val_edges2 = np.zeros(adj_label.shape)
    val_edges2[val_edges]=1
    val_edges2=species.T.dot(val_edges2)
    val_edges2= (val_edges2>0)& np.logical_not(forbidden)
    forbidden = forbidden + val_edges2
    
    val_edges_false2=np.zeros(adj_label.shape)
    val_edges_false2[val_edges_false]=1
    val_edges_false2=species.T.dot(val_edges_false2)
    val_edges_false2=(val_edges_false2>0) & np.logical_not(forbidden)
    forbidden = forbidden + val_edges_false2
    
    
    test_edges2 = np.zeros(adj_label.shape)
    test_edges2[test_edges]=1
    test_edges2=species.T.dot(test_edges2)
    test_edges2 = (test_edges2>0)& np.logical_not(forbidden)
    forbidden = forbidden + test_edges2
    
    
    test_edges_false2=np.zeros(adj_label.shape)
    test_edges_false2[test_edges_false]=1
    test_edges_false2=species.T.dot(test_edges_false2)
    test_edges_false2=(test_edges_false2>0)& np.logical_not(forbidden)
    
    n_sample = np.min([val_edges2.sum(),val_edges_false2.sum()])
    val_edges2 = np.where(val_edges2)
    val_edges_false2 = np.where(val_edges_false2)    
    i1=np.random.choice(range(val_edges2[0].shape[0]),n_sample,replace=False)
    i2= np.random.choice(range(val_edges_false2[0].shape[0]),n_sample,replace=False)
    val_edges2 = val_edges2[0][i1],val_edges2[1][i1]
    val_edges_false2 = val_edges_false2[0][i2],val_edges_false2[1][i2]
    
    n_sample = np.min([test_edges2.sum(),test_edges_false2.sum()])
    test_edges2 = np.where(test_edges2)
    test_edges_false2 = np.where(test_edges_false2)    
    i1=np.random.choice(range(test_edges2[0].shape[0]),n_sample,replace=False)
    i2= np.random.choice(range(test_edges_false2[0].shape[0]),n_sample,replace=False)
    test_edges2 = test_edges2[0][i1],test_edges2[1][i1]
    test_edges_false2 = test_edges_false2[0][i2],test_edges_false2[1][i2]
    
    return bipartite,val_edges2,val_edges_false2,test_edges2,test_edges_false2

