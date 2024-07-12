#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 24 17:06:04 2024

@author: mmip
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 19 14:15:55 2024

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

from preprocessing import *
from fair_model import *
from HSIC import *
import pandas
import scipy
import args


RES = pandas.DataFrame(columns=["AUC1_1","AUC2_1","AUC3_1","AUC4_1","AUC5_1","AUC6_1",
                  "AUC1_2","AUC2_2","AUC3_2","AUC4_2","AUC5_2","AUC6_2",
                  "AUC1_3","AUC2_3","AUC3_3","AUC4_3","AUC5_3","AUC6_3",
                  "AP1_1","AP2_1","AP3_1","AP4_1","AP5_1","AP6_1",
                  "AP1_2","AP2_2","AP3_2","AP4_2","AP5_2","AP6_2",
                  "AP1_3","AP2_3","AP3_3","AP4_3","AP5_3","AP6_3",
                  "SP1_1","SP2_1","SP3_1","SP4_1","SP5_1","SP6_1",
                  "HSIC1","HSIC2","HSIC3","HSIC4","HSIC5","HSIC6",
                  "pHSIC1","pHSIC2","pHSIC3","pHSIC4","pHSIC5","pHSIC6",
                  "cor1","cor2","cor3","cor4","cor5","cor6"],index=range(30))


def simulate_lbm(n1,n2,alpha,beta,P):
    W1 = np.random.choice(len(alpha),replace=True,p=alpha, size=n1)
    W2 = np.random.choice(len(beta) ,replace=True,p=beta , size=n2)
    proba = (P[W1].T[W2]).T
    M = np.random.binomial(1,proba)
    return W1,W2,M



alpha = (0.3,0.4,0.3)
beta = (0.2,0.4,0.4)
P = np.array([[0.95,0.80,0.5],
              [0.90,0.55,0.2],
              [0.7,0.25,0.06]])
n01=83
n02=306




W1,W2,bipartite_net = simulate_lbm(n01, n02, alpha, beta, P) 
plt.imshow(1-bipartite_net[np.argsort(W1),:][:,np.argsort(W2)],interpolation="none",cmap = "gray")



pbar = tqdm(range(30),desc = "k")
for K in pbar:
    W1,W2,bipartite_net = simulate_lbm(n01, n02, alpha, beta, P) 
    W3 = np.ones(len(W1))
    #W4 = np.random.binomial(1,np.array([0.1,1])[W2])
    W4 = np.random.binomial(1,np.array([0.1,0.4,0.9])[W2])
    
    n1 = 3000
    proba_obs1 = np.array([2/10,8/10])[W4]
    
    user_exp = np.random.exponential(21,size=n1)
    user_exp = np.round(user_exp)+1
    nb_obs = np.round(2*np.log(user_exp))
    species_index0 = np.random.randint(83,size=n1)
    species = np.zeros((species_index0.size, species_index0.max() + 1))
    species[np.arange(species_index0.size), species_index0] = 1
    
    net0 = np.zeros((n1,306))
    net_index=np.where(bipartite_net>0)
    for k in range(n1):
        possible = net_index[1][net_index[0]==species_index0[k]]
        proba_possible = proba_obs1[possible]
        proba_possible = proba_possible/sum(proba_possible)
        observed = np.random.choice(possible,int(nb_obs[k]),p=proba_possible)
        net0[k,observed] = 1
    
    SP = (species/species.sum(0)).T
    bipartite_obs = (SP@net0)
    
    adj0 = net0
    species01 = pandas.DataFrame(species.copy())
    features1 =species01.copy()
    features02 = np.eye(adj0.shape[1])
    features1 = sp.csr_matrix(features1) 
    species1 = sp.csr_matrix(species01) 
    features2 = sp.csr_matrix(features02) 
    adj = sp.csr_matrix(adj0) 
    features1 = sparse_to_tuple(features1.tocoo())
    species1 = sparse_to_tuple(species1.tocoo())
    features2 = sparse_to_tuple(features2.tocoo())
    adj_train, train_edges, val_edges, val_edges_false, test_edges, test_edges_false = mask_test_edges(adj0)
    adj_norm = preprocess_graph(adj_train)
    n=adj.shape[0]
    # Create Model
    pos_weight = float(adj.shape[0] * adj.shape[1] - adj.sum()) / adj.sum()
    norm = adj.shape[0] * adj.shape[1] / float((adj.shape[0] * adj.shape[1] - adj.sum()) * 2)
    adj_label = adj_train 
    adj_label = sparse_to_tuple(adj_label)
    adj_norm = torch.sparse.FloatTensor(torch.LongTensor(adj_norm[0].T), 
                                torch.FloatTensor(adj_norm[1]), 
                                torch.Size(adj_norm[2]))
    adj_label = torch.sparse.FloatTensor(torch.LongTensor(adj_label[0].T), 
                                torch.FloatTensor(adj_label[1]), 
                                torch.Size(adj_label[2]))
    
    
    features1 = torch.sparse.FloatTensor(torch.LongTensor(features1[0].T), 
                                torch.FloatTensor(features1[1]), 
                                torch.Size(features1[2]))
    features2 = torch.sparse.FloatTensor(torch.LongTensor(features2[0].T), 
                                torch.FloatTensor(features2[1]), 
                                torch.Size(features2[2]))
    
    species1 = torch.sparse.FloatTensor(torch.LongTensor(species1[0].T), 
                                torch.FloatTensor(species1[1]), 
                                torch.Size(species1[2]))
    
    weight_mask = adj_label.to_dense().view(-1) == 1
    weight_tensor = torch.ones(weight_mask.size(0)) 
    weight_tensor[weight_mask] = pos_weight
    
    ##########################################
    
    species_index =  np.array((np.where(species01))).T[:,1]
    
    #bipartite,val_edges2,val_edges_false2,test_edges2,test_edges_false2=mask_test_edges2(adj_label,species01.to_numpy(), val_edges, val_edges_false, test_edges, test_edges_false)
    bipartite,val_edges2,val_edges_false2,test_edges2,test_edges_false2=mask_test_edges3(adj_label,species01.to_numpy(),bipartite_net, val_edges, val_edges_false, test_edges, test_edges_false)
    
    pos_weight2 = (bipartite.shape[0]*bipartite.shape[1]-bipartite.sum())/(bipartite.sum())
    weight_tensor2 = torch.ones(bipartite.reshape(-1).shape[0]) 
    weight_tensor2[bipartite.reshape(-1)==1] = pos_weight2
    
    norm2 = bipartite.shape[0] * bipartite.shape[1] / float((bipartite.shape[0] *bipartite.shape[1] - bipartite.sum()) * 2)
    
    
        
        
    S0= torch.Tensor(user_exp).reshape(-1,1)
    S = (S0-S0.mean(0))/S0.std(0)
    
    
    model = VBGAE2(adj_norm,species_index)
    init_parameters(model)
    
    optimizer = Adam(model.parameters(), lr=args.learning_rate)
    
    # train model
    roclist = []
    loss_list= []
    
    
    #torch.manual_seed(1)
    for epoch in range(2*int(args.num_epoch)):
    
        A_pred,A_pred2,Z1,Z2,Z3 = model(features1,features2)
        optimizer.zero_grad()
        loss  = norm2*F.binary_cross_entropy(A_pred2.view(-1), torch.Tensor(bipartite).view(-1),weight = weight_tensor2)
        loss += norm*F.binary_cross_entropy(A_pred.view(-1), adj_label.to_dense().view(-1), weight = weight_tensor)
        kl_divergence = 0.5/ A_pred.size(0) *( (1 + 2*model.logstd1 - model.mean1**2 - torch.exp(model.logstd1)**2).sum(1).mean()+
                                              (1 + 2*model.logstd2 - model.mean2**2 - torch.exp(model.logstd2)**2).sum(1).mean())
        loss -= kl_divergence
        loss.backward()
        optimizer.step()
        
   
    A_pred_list=[(model(features1,features2)[0:2]) for j in range(10)]
    A_pred_list,A_pred_list2 = [A[0].detach().numpy() for A in A_pred_list],[A[1].detach().numpy() for A in A_pred_list]
    SP = (species01/species01.sum(0)).T.to_numpy()
    
    
    test_roc, test_ap = zip(*[(get_scores(test_edges, test_edges_false,torch.Tensor(A_pred))) for A_pred in A_pred_list])
    
    test_roc2, test_ap2 = zip(*[(get_scores(test_edges2, test_edges_false2,torch.Tensor(A_pred2))) for A_pred2 in A_pred_list2])

    A_pred_list3 = [(SP@A) for A in A_pred_list]
    test_roc3, test_ap3 = zip(*[(get_scores(test_edges2, test_edges_false2,torch.Tensor(A_pred3))) for A_pred3 in A_pred_list3])
    
    
    AUC1_1,AUC1_2,AUC1_3 = np.mean(test_roc),np.mean(test_roc2),np.mean(test_roc3)
    AP1_1,AP1_2,AP1_3 = np.mean(test_ap),np.mean(test_ap2),np.mean(test_ap3)

    
    TEST_EDGES2 = (np.concatenate((test_edges2[0], test_edges_false2[0])),
                   np.concatenate((test_edges2[1], test_edges_false2[1])))
    
    proba = (P[W1].T[W2]).T
    SP1_1 = np.mean([scipy.stats.spearmanr(proba[bipartite_obs==0].reshape(-1,1),
                          A_pred3[bipartite_obs==0].reshape(-1,1))[0] for A_pred3 in A_pred_list3])
    
    latent_space1= model.mean1
    cor1 = torch.linalg.norm(torch.corrcoef(torch.cat([latent_space1,S],axis=1).T)[-1,:-1]).item()

    stat1 = HSIC_stat(model.mean1,S)
    HSIC1 = stat1[0].item()
    pHSIC1=stats.gamma.sf(stat1[0].item()*n, stat1[3].item(), scale=stat1[4].item())
    

    model2 = VBGAE2(adj_norm,species_index)
    init_parameters(model2)

    optimizer = Adam(model2.parameters(), lr=args.learning_rate)

   
    for epoch in range(2*int(args.num_epoch)):

        A_pred,A_pred2,Z1,Z2,Z3 = model2(features1,features2)
        optimizer.zero_grad()
        loss  = norm2*F.binary_cross_entropy(A_pred2.view(-1), torch.Tensor(bipartite).view(-1),weight = weight_tensor2)
        loss += norm*F.binary_cross_entropy(A_pred.view(-1), adj_label.to_dense().view(-1), weight = weight_tensor)
        kl_divergence = 0.5/ A_pred.size(0) *( (1 + 2*model2.logstd1 - model2.mean1**2 - torch.exp(model2.logstd1)**2).sum(1).mean()+
                                              (1 + 2*model2.logstd2 - model2.mean2**2 - torch.exp(model2.logstd2)**2).sum(1).mean())
        loss -= kl_divergence
        #independance =torch.log(RFF_HSIC(model2.mean1,S))
        independance = 0.25*n*RFF_HSIC(model2.mean1,S)
        loss += independance

        loss.backward()
        optimizer.step()
    
    A_pred_list=[(model2(features1,features2)[0:2]) for j in range(10)]
    A_pred_list,A_pred_list2 = [A[0].detach().numpy() for A in A_pred_list],[A[1].detach().numpy() for A in A_pred_list]
    
    
    test_roc, test_ap = zip(*[(get_scores(test_edges, test_edges_false,torch.Tensor(A_pred))) for A_pred in A_pred_list])
    
    test_roc2, test_ap2 = zip(*[(get_scores(test_edges2, test_edges_false2,torch.Tensor(A_pred2))) for A_pred2 in A_pred_list2])

    A_pred_list3 = [(SP@A) for A in A_pred_list]
    test_roc3, test_ap3 = zip(*[(get_scores(test_edges2, test_edges_false2,torch.Tensor(A_pred3))) for A_pred3 in A_pred_list3])
    
    
    AUC2_1,AUC2_2,AUC2_3 = np.mean(test_roc),np.mean(test_roc2),np.mean(test_roc3)
    AP2_1,AP2_2,AP2_3 = np.mean(test_ap),np.mean(test_ap2),np.mean(test_ap3)

   
    SP2_1 = np.mean([scipy.stats.spearmanr(proba[bipartite_obs==0].reshape(-1,1),
                          A_pred3[bipartite_obs==0].reshape(-1,1))[0] for A_pred3 in A_pred_list3])
    
    
    stat1 = HSIC_stat(model2.mean1,S)
    HSIC2 = stat1[0].item()
    pHSIC2=stats.gamma.sf(stat1[0].item()*n, stat1[3].item(), scale=stat1[4].item())
    latent_space1= model2.mean1
    cor2 = torch.linalg.norm(torch.corrcoef(torch.cat([latent_space1,S0],axis=1).T)[-1,:-1]).item()


    model3 = VBGAE3(adj_norm,species_index,2)
    init_parameters(model3)
    
    optimizer = Adam(model3.parameters(), lr=args.learning_rate)
    
    # train model
    roclist = []
    loss_list= []
    
    
    #torch.manual_seed(1)
    for epoch in range(2*int(args.num_epoch)):
    
        A_pred,A_pred2,Z1,Z2,Z3 = model3(features1,features2)
        optimizer.zero_grad()
        loss  = norm2*F.binary_cross_entropy(A_pred2.view(-1), torch.Tensor(bipartite).view(-1),weight = weight_tensor2)
        loss += norm*F.binary_cross_entropy(A_pred.view(-1), adj_label.to_dense().view(-1), weight = weight_tensor)
        kl_divergence = 0.5/ A_pred.size(0) *( (1 + 2*model3.logstd1 - model3.mean1**2 - torch.exp(model3.logstd1)**2).sum(1).mean()+
                                              (1 + 2*model3.logstd2 - model3.mean2**2 - torch.exp(model3.logstd2)**2).sum(1).mean())
        loss -= kl_divergence
        loss.backward()
        optimizer.step()
        
   
    A_pred_list=[(model3(features1,features2)[0:2]) for j in range(10)]
    A_pred_list,A_pred_list2 = [A[0].detach().numpy() for A in A_pred_list],[A[1].detach().numpy() for A in A_pred_list]
    
    
    test_roc, test_ap = zip(*[(get_scores(test_edges, test_edges_false,torch.Tensor(A_pred))) for A_pred in A_pred_list])
    
    test_roc2, test_ap2 = zip(*[(get_scores(test_edges2, test_edges_false2,torch.Tensor(A_pred2))) for A_pred2 in A_pred_list2])

    A_pred_list3 = [(SP@A) for A in A_pred_list]
    test_roc3, test_ap3 = zip(*[(get_scores(test_edges2, test_edges_false2,torch.Tensor(A_pred3))) for A_pred3 in A_pred_list3])
    
    
    AUC3_1,AUC3_2,AUC3_3 = np.mean(test_roc),np.mean(test_roc2),np.mean(test_roc3)
    AP3_1,AP3_2,AP3_3 = np.mean(test_ap),np.mean(test_ap2),np.mean(test_ap3)

    
   
    SP3_1 = np.mean([scipy.stats.spearmanr(proba[bipartite_obs==0].reshape(-1,1),
                          A_pred3[bipartite_obs==0].reshape(-1,1))[0] for A_pred3 in A_pred_list3])
    
    latent_space1= model3.mean1
    cor3 = torch.linalg.norm(torch.corrcoef(torch.cat([latent_space1,S],axis=1).T)[-1,:-1]).item()

    stat1 = HSIC_stat(model3.mean1,S)
    HSIC3 = stat1[0].item()
    pHSIC3=stats.gamma.sf(stat1[0].item()*n, stat1[3].item(), scale=stat1[4].item())
    

    model4 = VBGAE3(adj_norm,species_index,2)
    init_parameters(model4)

    optimizer = Adam(model4.parameters(), lr=args.learning_rate)

   
    for epoch in range(2*int(args.num_epoch)):

        A_pred,A_pred2,Z1,Z2,Z3 = model4(features1,features2)
        optimizer.zero_grad()
        loss  = norm2*F.binary_cross_entropy(A_pred2.view(-1), torch.Tensor(bipartite).view(-1),weight = weight_tensor2)
        loss += norm*F.binary_cross_entropy(A_pred.view(-1), adj_label.to_dense().view(-1), weight = weight_tensor)
        kl_divergence = 0.5/ A_pred.size(0) *( (1 + 2*model4.logstd1 - model4.mean1**2 - torch.exp(model4.logstd1)**2).sum(1).mean()+
                                              (1 + 2*model4.logstd2 - model4.mean2**2 - torch.exp(model4.logstd2)**2).sum(1).mean())
        loss -= kl_divergence
        #independance =torch.log(RFF_HSIC(model2.mean1,S))
        independance = 0.25*n*RFF_HSIC(model4.mean1,S)
        loss += independance

        loss.backward()
        optimizer.step()
    
    A_pred_list=[(model4(features1,features2)[0:2]) for j in range(10)]
    A_pred_list,A_pred_list2 = [A[0].detach().numpy() for A in A_pred_list],[A[1].detach().numpy() for A in A_pred_list]
    
    
    test_roc, test_ap = zip(*[(get_scores(test_edges, test_edges_false,torch.Tensor(A_pred))) for A_pred in A_pred_list])
    
    test_roc2, test_ap2 = zip(*[(get_scores(test_edges2, test_edges_false2,torch.Tensor(A_pred2))) for A_pred2 in A_pred_list2])

    A_pred_list3 = [(SP@A) for A in A_pred_list]
    test_roc3, test_ap3 = zip(*[(get_scores(test_edges2, test_edges_false2,torch.Tensor(A_pred3))) for A_pred3 in A_pred_list3])
    
    
    AUC4_1,AUC4_2,AUC4_3 = np.mean(test_roc),np.mean(test_roc2),np.mean(test_roc3)
    AP4_1,AP4_2,AP4_3 = np.mean(test_ap),np.mean(test_ap2),np.mean(test_ap3)

   
    SP4_1 = np.mean([scipy.stats.spearmanr(proba[bipartite_obs==0].reshape(-1,1),
                          A_pred3[bipartite_obs==0].reshape(-1,1))[0] for A_pred3 in A_pred_list3])
    
    
    stat1 = HSIC_stat(model4.mean1,S)
    HSIC4 = stat1[0].item()
    pHSIC4=stats.gamma.sf(stat1[0].item()*n, stat1[3].item(), scale=stat1[4].item())
    latent_space1= model4.mean1
    cor4 = torch.linalg.norm(torch.corrcoef(torch.cat([latent_space1,S0],axis=1).T)[-1,:-1]).item()

#############

    model5 =  VBGAE2(adj_norm,species_index)
    adv = Adversary(1)
    init_parameters(model5)
    
    optimizer = Adam(model5.parameters(), lr=args.learning_rate)
    adv_optimizer = Adam(adv.parameters(),lr = 0.01)
    # train model

    
    #torch.manual_seed(1)
    for epoch in range(400):
        #train model
        A_pred,A_pred2,Z1,Z2,Z3 = model5(features1,features2)
        optimizer.zero_grad()
        loss  = norm2*F.binary_cross_entropy(A_pred2.view(-1), torch.Tensor(bipartite).view(-1),weight = weight_tensor2)
        loss += norm*F.binary_cross_entropy(A_pred.view(-1), adj_label.to_dense().view(-1), weight = weight_tensor)
        kl_divergence = 0.5/ A_pred.size(0) *( (1 + 2*model5.logstd1 - model5.mean1**2 - torch.exp(model5.logstd1)**2).sum(1).mean()+
                                              (1 + 2*model5.logstd2 - model5.mean2**2 - torch.exp(model5.logstd2)**2).sum(1).mean())
        loss -= kl_divergence
        loss.backward()
        optimizer.step()

    for epoch in range(1000):
        #train adv
            adv_optimizer.zero_grad()
            A_pred,A_pred2,Z1,Z2,Z3 = model5(features1,features2)
            Z1 = Z1.detach()
            s_hat = adv(Z1)
            adv_loss = -torch.abs(torch.corrcoef(torch.cat([S,s_hat],axis=1).T)[0,1])
            adv_loss.backward()
            adv_optimizer.step()


    adv_loss_list = []
    for epoch in range(2*int(args.num_epoch)):
        #train adv
        adv_optimizer.zero_grad()
        A_pred,A_pred2,Z1,Z2,Z3  = model(features1,features2)
        Z1 = Z1.detach()
        s_hat = adv(Z1)
        adv_loss = -torch.abs(torch.corrcoef(torch.cat([S,s_hat],axis=1).T)[0,1])
        adv_loss.backward()
        adv_optimizer.step()
        
        #train adv
        A_pred,A_pred2,Z1,Z2,Z3 = model5(features1,features2)
        optimizer.zero_grad()
        loss  = norm2*F.binary_cross_entropy(A_pred2.view(-1), torch.Tensor(bipartite).view(-1),weight = weight_tensor2)
        loss += norm*F.binary_cross_entropy(A_pred.view(-1), adj_label.to_dense().view(-1), weight = weight_tensor)
        kl_divergence = 0.5/ A_pred.size(0) *( (1 + 2*model5.logstd1 - model5.mean1**2 - torch.exp(model5.logstd1)**2).sum(1).mean()+
                                              (1 + 2*model5.logstd2 - model5.mean2**2 - torch.exp(model5.logstd2)**2).sum(1).mean())
        loss -= kl_divergence
        s_hat = adv(Z1)
        adv_loss = -torch.abs(torch.corrcoef(torch.cat([S,s_hat],axis=1).T)[0,1])
        adv_loss_list.append(adv_loss.item())
        loss -= adv_loss#*torch.tensor(epoch)
        loss.backward()
        optimizer.step()

    A_pred_list=[(model5(features1,features2)[0:2]) for j in range(10)]
    A_pred_list,A_pred_list2 = [A[0].detach().numpy() for A in A_pred_list],[A[1].detach().numpy() for A in A_pred_list]
    
    
    test_roc, test_ap = zip(*[(get_scores(test_edges, test_edges_false,torch.Tensor(A_pred))) for A_pred in A_pred_list])
    
    test_roc2, test_ap2 = zip(*[(get_scores(test_edges2, test_edges_false2,torch.Tensor(A_pred2))) for A_pred2 in A_pred_list2])

    A_pred_list3 = [(SP@A) for A in A_pred_list]
    test_roc3, test_ap3 = zip(*[(get_scores(test_edges2, test_edges_false2,torch.Tensor(A_pred3))) for A_pred3 in A_pred_list3])
    
    
    AUC5_1,AUC5_2,AUC5_3 = np.mean(test_roc),np.mean(test_roc2),np.mean(test_roc3)
    AP5_1,AP5_2,AP5_3 = np.mean(test_ap),np.mean(test_ap2),np.mean(test_ap3)

   
    SP5_1 = np.mean([scipy.stats.spearmanr(proba[bipartite_obs==0].reshape(-1,1),
                          A_pred3[bipartite_obs==0].reshape(-1,1))[0] for A_pred3 in A_pred_list3])
    
    
    stat1 = HSIC_stat(model5.mean1,S)
    HSIC5 = stat1[0].item()
    pHSIC5=stats.gamma.sf(stat1[0].item()*n, stat1[3].item(), scale=stat1[4].item())
    latent_space1= model5.mean1
    cor5 = torch.linalg.norm(torch.corrcoef(torch.cat([latent_space1,S0],axis=1).T)[-1,:-1]).item()
    

    model6 =  VBGAE3(adj_norm,species_index,2)
    adv = Adversary(1)
    init_parameters(model6)
    
    optimizer = Adam(model6.parameters(), lr=args.learning_rate)
    adv_optimizer = Adam(adv.parameters(),lr = 0.01)
    # train model

    
    #torch.manual_seed(1)
    for epoch in range(400):
        #train model
        A_pred,A_pred2,Z1,Z2,Z3 = model6(features1,features2)
        optimizer.zero_grad()
        loss  = norm2*F.binary_cross_entropy(A_pred2.view(-1), torch.Tensor(bipartite).view(-1),weight = weight_tensor2)
        loss += norm*F.binary_cross_entropy(A_pred.view(-1), adj_label.to_dense().view(-1), weight = weight_tensor)
        kl_divergence = 0.5/ A_pred.size(0) *( (1 + 2*model6.logstd1 - model6.mean1**2 - torch.exp(model6.logstd1)**2).sum(1).mean()+
                                              (1 + 2*model6.logstd2 - model6.mean2**2 - torch.exp(model6.logstd2)**2).sum(1).mean())
        loss -= kl_divergence
        loss.backward()
        optimizer.step()

    for epoch in range(1000):
        #train adv
            adv_optimizer.zero_grad()
            A_pred,A_pred2,Z1,Z2,Z3 = model6(features1,features2)
            Z1 = Z1.detach()
            s_hat = adv(Z1)
            adv_loss = -torch.abs(torch.corrcoef(torch.cat([S,s_hat],axis=1).T)[0,1])
            adv_loss.backward()
            adv_optimizer.step()



    for epoch in range(2*int(args.num_epoch)):
        #train adv
        adv_optimizer.zero_grad()
        A_pred,A_pred2,Z1,Z2,Z3  = model(features1,features2)
        Z1 = Z1.detach()
        s_hat = adv(Z1)
        adv_loss = -torch.abs(torch.corrcoef(torch.cat([S,s_hat],axis=1).T)[0,1])
        adv_loss.backward()
        adv_optimizer.step()
        
        #train adv
        A_pred,A_pred2,Z1,Z2,Z3 = model6(features1,features2)
        optimizer.zero_grad()
        loss  = norm2*F.binary_cross_entropy(A_pred2.view(-1), torch.Tensor(bipartite).view(-1),weight = weight_tensor2)
        loss += norm*F.binary_cross_entropy(A_pred.view(-1), adj_label.to_dense().view(-1), weight = weight_tensor)
        kl_divergence = 0.5/ A_pred.size(0) *( (1 + 2*model6.logstd1 - model6.mean1**2 - torch.exp(model6.logstd1)**2).sum(1).mean()+
                                              (1 + 2*model6.logstd2 - model6.mean2**2 - torch.exp(model6.logstd2)**2).sum(1).mean())
        loss -= kl_divergence
        s_hat = adv(Z1)
        adv_loss = -torch.abs(torch.corrcoef(torch.cat([S,s_hat],axis=1).T)[0,1])
        loss -= adv_loss#*torch.tensor(epoch)
        loss.backward()
        optimizer.step()

    A_pred_list=[(model6(features1,features2)[0:2]) for j in range(10)]
    A_pred_list,A_pred_list2 = [A[0].detach().numpy() for A in A_pred_list],[A[1].detach().numpy() for A in A_pred_list]
    
    
    test_roc, test_ap = zip(*[(get_scores(test_edges, test_edges_false,torch.Tensor(A_pred))) for A_pred in A_pred_list])
    
    test_roc2, test_ap2 = zip(*[(get_scores(test_edges2, test_edges_false2,torch.Tensor(A_pred2))) for A_pred2 in A_pred_list2])

    A_pred_list3 = [(SP@A) for A in A_pred_list]
    test_roc3, test_ap3 = zip(*[(get_scores(test_edges2, test_edges_false2,torch.Tensor(A_pred3))) for A_pred3 in A_pred_list3])
    
    
    AUC6_1,AUC6_2,AUC6_3 = np.mean(test_roc),np.mean(test_roc2),np.mean(test_roc3)
    AP6_1,AP6_2,AP6_3 = np.mean(test_ap),np.mean(test_ap2),np.mean(test_ap3)

   
    SP6_1 = np.mean([scipy.stats.spearmanr(proba[bipartite_obs==0].reshape(-1,1),
                          A_pred3[bipartite_obs==0].reshape(-1,1))[0] for A_pred3 in A_pred_list3])
    
    
    stat1 = HSIC_stat(model6.mean1,S)
    HSIC6 = stat1[0].item()
    pHSIC6=stats.gamma.sf(stat1[0].item()*n, stat1[3].item(), scale=stat1[4].item())
    latent_space1= model6.mean1
    cor6 = torch.linalg.norm(torch.corrcoef(torch.cat([latent_space1,S0],axis=1).T)[-1,:-1]).item()
    
    





    RES.iloc[K,:] =  [AUC1_1,AUC2_1,AUC3_1,AUC4_1,AUC5_1,AUC6_1,
                      AUC1_2,AUC2_2,AUC3_2,AUC4_2,AUC5_2,AUC6_2,
                      AUC1_3,AUC2_3,AUC3_3,AUC4_3,AUC5_3,AUC6_3,
                      AP1_1,AP2_1,AP3_1,AP4_1,AP5_1,AP6_1,
                      AP1_2,AP2_2,AP3_2,AP4_2,AP5_2,AP6_2,
                      AP1_3,AP2_3,AP3_3,AP4_3,AP5_3,AP6_3,
                      SP1_1,SP2_1,SP3_1,SP4_1,SP5_1,SP6_1,
                      HSIC1,HSIC2,HSIC3,HSIC4,HSIC5,HSIC6,
                      pHSIC1,pHSIC2,pHSIC3,pHSIC4,pHSIC5,pHSIC6,
                      cor1,cor2,cor3,cor4,cor5,cor6]
    
    print(RES.iloc[K,:])
    print("#####")
    print(np.round(RES.mean(0),3))
    results.to_csv("results.csv")
