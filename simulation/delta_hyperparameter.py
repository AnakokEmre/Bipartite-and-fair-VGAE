#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 15 11:15:46 2023

@author: mmip
"""

import torch
import torch.nn.functional as F
from torch.optim import Adam
import scipy.sparse as sp
import numpy as np
import os
import time
import matplotlib.pyplot as plt
from tqdm import tqdm
import matplotlib.patches as mpatches
from preprocessing import *
from model import *
from HSIC import *
import pandas



n1=1000
n2=100

features01 = np.eye(n1)
features02 = np.eye(n2)

features1 = sp.csr_matrix(features01) 
features2 = sp.csr_matrix(features02) 


features1 = sparse_to_tuple(features1.tocoo())
features2 = sparse_to_tuple(features2.tocoo())


features1 = torch.sparse.FloatTensor(torch.LongTensor(features1[0].T), 
                        torch.FloatTensor(features1[1]), 
                        torch.Size(features1[2]))
features2 = torch.sparse.FloatTensor(torch.LongTensor(features2[0].T), 
                        torch.FloatTensor(features2[1]), 
                        torch.Size(features2[2]))

delta_hyperparameter_result = pandas.DataFrame(columns = ["AUC","AP","HSIC","p_value","#0.05","cov_1","delta","k"])
delta_hyperparameter = [0,10,100,200,500,1000,2000]

for k in range(100):
    print(10*"#")
    print(k)
    print(10*"#")
    
    z = np.random.normal(size=(n1,1))
    s = np.random.normal(size=(n1,1))
    S = torch.Tensor(s)
    z1 = np.concatenate([s,z],axis=1)
    z2 =  np.random.normal(loc = 0,size=(n2,2))
    

    adj0=torch.bernoulli(distance_decode(torch.Tensor(z1),torch.Tensor(z2)))
    adj_train, train_edges, val_edges, val_edges_false, test_edges, test_edges_false = mask_test_edges(adj0)
    adj_norm = preprocess_graph(adj_train)
    adj = sp.csr_matrix(adj0) 

    n=adj.shape[0]
    # Create Model
    pos_weight = float(n1 * n2 - adj.sum()) / adj.sum()
    norm = n1 * n2 / float((n1 * n2 - adj.sum()) * 2)
    
    
    adj_label = adj_train 
    adj_label = sparse_to_tuple(adj_label)
    
    
    
    adj_norm = torch.sparse.FloatTensor(torch.LongTensor(adj_norm[0].T), 
                                torch.FloatTensor(adj_norm[1]), 
                                torch.Size(adj_norm[2]))
    adj_label = torch.sparse.FloatTensor(torch.LongTensor(adj_label[0].T), 
                                torch.FloatTensor(adj_label[1]), 
                                torch.Size(adj_label[2]))
    
    
  
    
    
    weight_mask = adj_label.to_dense().view(-1) == 1
    weight_tensor = torch.ones(weight_mask.size(0)) 
    weight_tensor[weight_mask] = pos_weight
    
    for delta in delta_hyperparameter :

        list_model2 =  [VBGAE_GRDPG(adj_norm,n1,n2,1) for k in range(10)]
        list_val_roc2 = []
        for model2 in list_model2:
            init_parameters(model2)
            optimizer = Adam(model2.parameters(), lr=args.learning_rate*2)
            
            pbar = tqdm(range(1000),desc = "Training Epochs")
            for epoch in pbar:
                t = time.time()
            
                A_pred,Z1,Z2 = model2(features1,features2)
                optimizer.zero_grad()
                loss = norm*F.binary_cross_entropy(A_pred.view(-1), adj_label.to_dense().view(-1), weight = weight_tensor)
            
                kl_divergence = 0.5/ A_pred.size(0) *( (1 + 2*model2.logstd1 - model2.mean1**2 - torch.exp(model2.logstd1)**2).sum(1).mean()+
                                                      (1 + 2*model2.logstd2 - model2.mean2**2 - torch.exp(model2.logstd2)**2).sum(1).mean())
                loss -= kl_divergence
                independance =delta*(RFF_HSIC(model2.mean1,S))#0.02
            
                loss += independance
                loss.backward()
                optimizer.step()
                
            
                val_roc, val_ap = get_scores(val_edges, val_edges_false, A_pred)
                pbar.set_postfix({"train_loss=": "{:.5f}".format(loss.item()),
                                  'val_roc=': val_roc,'log_HSIC =': independance.item()})
            list_val_roc2.append(val_roc)
    
        best_model2 = list_model2[np.argmax(list_val_roc2)]
        latent_space1=best_model2.mean1
        latent_space2=best_model2.mean2
        A_pred,Z1,Z2 = best_model2(features1,features2)
        test_roc, test_ap = get_scores(test_edges, test_edges_false, A_pred)
    
        stat2 = HSIC_stat(latent_space1,S)
        zut=stats.gamma.sf(stat2[0].item()*n, stat2[3].item(), scale=stat2[4].item())
        cov2 = torch.linalg.norm(torch.corrcoef(torch.cat([latent_space1,torch.Tensor(z1)],axis=1).T)[2,:2]).item()
    
        delta_hyperparameter_result.loc[len(delta_hyperparameter_result)] = [test_roc,test_ap,stat2[0].detach().numpy() ,zut,zut<0.05,cov2,delta,k]
        print(delta_hyperparameter_result.iloc[len(delta_hyperparameter_result)-1])
        delta_hyperparameter_result.to_csv("delta_hyperparameter_result.csv")
    


