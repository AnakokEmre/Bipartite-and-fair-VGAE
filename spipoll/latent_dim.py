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

#os.environ['CUDA_VISIBLE_DEVICES'] = ""

import pandas


adj0=pandas.read_csv("net.csv",header=0,sep="\t").to_numpy(dtype=float)
features01 = pandas.read_csv("features.csv",header=0,sep="\t")
species01 = pandas.read_csv("species.csv",header=0,sep="\t")

#mean_Temperature,std_Temperature = features01["Temperature"].mean(),features01["Temperature"].std()
mean_Temperature,std_Temperature = features01["Temperature"].mean(),features01["Temperature"].std()

features1 = species01.copy()
features1["Temperature"] = (features01["Temperature"]-mean_Temperature)/std_Temperature


features02 = np.eye(adj0.shape[1])

features1 = sp.csr_matrix(features1) 
species1 = sp.csr_matrix(species01) 

features2 = sp.csr_matrix(features02) 


features1 = sparse_to_tuple(features1.tocoo())
species1 = sparse_to_tuple(species1.tocoo())
features2 = sparse_to_tuple(features2.tocoo())


# Some preprocessing

S0 = torch.Tensor(pandas.read_csv("S.csv",sep="\t").to_numpy())
S = S0.clone()
S[:,0] = torch.log10(S0[:,0])
S = (S0-S0.mean(0))/S0.std(0)

import args

SP = (species01/species01.sum(0)).T.to_numpy()





result = pandas.DataFrame(columns = ["AUC","AP",
                                     "AUC2","AP2",
                                     "AUC3","AP3",
                                     "HSIC","p_value","#0.05","cor2","Z_dim"],
                          index = range(100))



fair_result = pandas.DataFrame(columns = ["AUC","AP",
                                     "AUC2","AP2",
                                     "AUC3","AP3",
                                     "HSIC","p_value","#0.05","cor2","Z_dim"],
                          index = range(100))

torch.manual_seed(1)

n1 = adj.shape[0]

row_index = 0

for Z_dim in range(1,11):
    args.hidden2_dim1 = Z_dim
    args.hidden2_dim2 = args.hidden2_dim1

    for k in range(10):
        adj = sp.csr_matrix(adj0) 
        adj_train, train_edges, val_edges, val_edges_false, test_edges, test_edges_false = mask_test_edges(adj0)
        
        adj_norm = preprocess_graph(adj_train)
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
        weight_mask = adj_label.to_dense().view(-1) == 1
        weight_tensor = torch.ones(weight_mask.size(0)) 
        weight_tensor[weight_mask] = pos_weight
        bipartite,val_edges2,val_edges_false2,test_edges2,test_edges_false2=mask_test_edges2(adj_label,species01.to_numpy(), val_edges, val_edges_false, test_edges, test_edges_false)
        pos_weight2 = (bipartite.shape[0]*bipartite.shape[1]-bipartite.sum())/(bipartite.sum())
        weight_tensor2 = torch.ones(bipartite.reshape(-1).shape[0]) 
        weight_tensor2[bipartite.reshape(-1)==1] = pos_weight2
        norm2 = bipartite.shape[0] * bipartite.shape[1] / float((bipartite.shape[0] *bipartite.shape[1] - bipartite.sum()) * 2)
        
        
        model = VBGAE2(adj_norm,species_index) 
        init_parameters(model)
        optimizer = Adam(model.parameters(), lr=args.learning_rate)
        # train model
      
        pbar = tqdm(range(int(args.num_epoch)),desc = "Training Epochs")
        for epoch in pbar:
            t = time.time()
    
            A_pred,A_pred2,Z1,Z2,Z3 = model(features1,features2)
            optimizer.zero_grad()
            loss  = norm2*F.binary_cross_entropy(A_pred2.view(-1), torch.Tensor(bipartite).view(-1),weight = weight_tensor2)
            loss += norm*F.binary_cross_entropy(A_pred.view(-1), adj_label.to_dense().view(-1), weight = weight_tensor)
            kl_divergence = 0.5/ A_pred.size(0) *( (1 + 2*model.logstd1 - model.mean1**2 - torch.exp(model.logstd1)**2).sum(1).mean()+
                                                  (1 + 2*model.logstd2 - model.mean2**2 - torch.exp(model.logstd2)**2).sum(1).mean())
            loss -= kl_divergence
            loss.backward()
            optimizer.step()
            
    
            val_roc, val_ap = get_scores(val_edges, val_edges_false, A_pred)
            val_roc2, val_ap2 = get_scores(val_edges2, val_edges_false2, A_pred2)
            
    
            pbar.set_postfix({"train_loss=": "{:.5f}".format(loss.item()),
                              'val_roc=': val_roc,
                              "val_roc2=": "{:.5f}".format(val_roc2)})
            
        latent_space1=model.mean1
        latent_space2=model.mean2
        A_pred,A_pred2,Z1,Z2,Z3 = model(features1,features2)
        A_pred3 = (SP@A_pred.detach().numpy())
    
    
        test_roc, test_ap = get_scores(test_edges, test_edges_false, A_pred)
        test_roc2, test_ap2 = get_scores(test_edges2, test_edges_false2, A_pred2)
        test_roc3, test_ap3= get_scores(test_edges2, test_edges_false2,torch.Tensor(A_pred3))
       
        
        cor2 = torch.linalg.norm(torch.corrcoef(torch.cat([latent_space1,S],axis=1).T)[-1,:-1]).item()
        stat1 = HSIC_stat(model.mean1,S)
        p005=stats.gamma.sf(stat1[0].item()*n1, stat1[3].item(), scale=stat1[4].item())
        
        result.iloc[row_index] = [test_roc,test_ap,
                          test_roc2, test_ap2,
                          test_roc3, test_ap3,
                          stat1[0].item() ,p005,p005<0.05,cor2,Z_dim]
        print(result.iloc[row_index])
        result.to_csv("spipoll_latent_dim.csv")
        
        model2 = VBGAE2(adj_norm,species_index)
        
        init_parameters(model2)
        optimizer = Adam(model2.parameters(), lr=args.learning_rate)
        # train model
      
        pbar = tqdm(range(int(args.num_epoch)),desc = "Training Epochs")
        for epoch in pbar:
            t = time.time()
    
            A_pred,A_pred2,Z1,Z2,Z3 = model2(features1,features2)
            optimizer.zero_grad()
            loss  = norm2*F.binary_cross_entropy(A_pred2.view(-1), torch.Tensor(bipartite).view(-1),weight = weight_tensor2)
            loss += norm*F.binary_cross_entropy(A_pred.view(-1), adj_label.to_dense().view(-1), weight = weight_tensor)
            kl_divergence = 0.5/ A_pred.size(0) *( (1 + 2*model2.logstd1 - model2.mean1**2 - torch.exp(model2.logstd1)**2).sum(1).mean()+
                                                  (1 + 2*model2.logstd2 - model2.mean2**2 - torch.exp(model2.logstd2)**2).sum(1).mean())
            loss -= kl_divergence
            #independance =torch.log(RFF_HSIC(model2.mean1,S))
            independance = n*RFF_HSIC(model2.mean1,S)
            loss += independance
    
            loss.backward()
            optimizer.step()
            
    
            val_roc, val_ap = get_scores(val_edges, val_edges_false, A_pred)
            val_roc2, val_ap2 = get_scores(val_edges2, val_edges_false2, A_pred2)
            
    
            pbar.set_postfix({"train_loss=": "{:.5f}".format(loss.item()),
                              'val_roc=': val_roc,
                              "val_roc2=": "{:.5f}".format(val_roc2) ,
                             "HSIC=": "{:.5f}".format(independance)})
            
        
        latent_space1= model2.mean1
        latent_space2= model2.mean2
        A_pred,A_pred2,Z1,Z2,Z3 = model2(features1,features2)
        A_pred3 = (SP@A_pred.detach().numpy())
    
    
        test_roc, test_ap = get_scores(test_edges, test_edges_false, A_pred)
        test_roc2, test_ap2 = get_scores(test_edges2, test_edges_false2, A_pred2)
        test_roc3, test_ap3= get_scores(test_edges2, test_edges_false2,torch.Tensor(A_pred3))
       
            
        cor2 = torch.linalg.norm(torch.corrcoef(torch.cat([latent_space1,S],axis=1).T)[-1,:-1]).item()
        stat1 = HSIC_stat(model2.mean1,S)
        p005=stats.gamma.sf(stat1[0].item()*n1, stat1[3].item(), scale=stat1[4].item())
        
        fair_result.iloc[row_index] = [test_roc,test_ap,
                          test_roc2, test_ap2,
                          test_roc3, test_ap3,
                          stat1[0].detach().numpy() ,p005,p005<0.05,cor2,Z_dim]
        print(fair_result.iloc[row_index])
        fair_result.to_csv("spipoll_fair_latent_dim.csv")
        row_index+=1



print(result)

from scipy import stats
confidence_interval = 0.95
t_confidence_interval = stats.t.ppf((1 + confidence_interval) / 2, result.groupby("Z_dim")["AUC"].count() - 1)

plt.errorbar(np.arange(1,11),
             result.groupby("Z_dim")["AUC"].mean(),
             yerr=t_confidence_interval*result.groupby("Z_dim")["AUC"].std()/np.sqrt(result.groupby("Z_dim")["AUC"].count()),
             fmt = "-o",
             capsize = 5,
             label = "BGVAE")

plt.errorbar(np.arange(1,11),
             fair_result.groupby("Z_dim")["AUC"].mean(),
             yerr=t_confidence_interval*fair_result.groupby("Z_dim")["AUC"].std()/np.sqrt(fair_result.groupby("Z_dim")["AUC"].count()),
             fmt = "-o",
             capsize = 5,
             label = "fair BVGAE")
plt.xlabel("Latent space dimension")
plt.xticks(np.arange(1,11))
plt.yticks(np.arange(0.4,1,0.1))
plt.ylabel("AUC1")
plt.legend()








