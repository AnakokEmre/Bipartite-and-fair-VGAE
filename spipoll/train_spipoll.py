"""
Created on May 2023
Original code by Daehan Kim
GitHub Repository https://github.com/DaehanKim/vgae_pytorch/blob/master/LICENSE

Modified by : Emre Anakok

Description : Modified the original code to adapt the VGAE to the bipartite case.
Copyright (c) 
Licence : MIT Licence
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


adj0=pandas.read_csv("data/net.csv",header=0,sep="\t").to_numpy(dtype=float)
features01 = pandas.read_csv("data/features.csv",header=0,sep="\t")
species01 = pandas.read_csv("data/species.csv",header=0,sep="\t")

mean_Temperature,std_Temperature = features01["Temperature"].mean(),features01["Temperature"].std()

features1 = species01.copy()
features1["Temperature"] = (features01["Temperature"]-mean_Temperature)/std_Temperature


features02 = np.eye(adj0.shape[1])

features1 = sp.csr_matrix(features1) 
species1 = sp.csr_matrix(species01) 

features2 = sp.csr_matrix(features02) 


adj = sp.csr_matrix(adj0) 
features1 = sparse_to_tuple(features1.tocoo())
species1 = sparse_to_tuple(species1.tocoo())
features2 = sparse_to_tuple(features2.tocoo())




adj_train, train_edges, val_edges, val_edges_false, test_edges, test_edges_false = mask_test_edges(adj0)

# Some preprocessing
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

bipartite,val_edges2,val_edges_false2,test_edges2,test_edges_false2=mask_test_edges2(adj_label,species01.to_numpy(), val_edges, val_edges_false, test_edges, test_edges_false)

pos_weight2 = (bipartite.shape[0]*bipartite.shape[1]-bipartite.sum())/(bipartite.sum())
weight_tensor2 = torch.ones(bipartite.reshape(-1).shape[0]) 
weight_tensor2[bipartite.reshape(-1)==1] = pos_weight2

norm2 = bipartite.shape[0] * bipartite.shape[1] / float((bipartite.shape[0] *bipartite.shape[1] - bipartite.sum()) * 2)



S0 = torch.Tensor(pandas.read_csv("data/S.csv",sep="\t").to_numpy())
S = S0.clone()
S[:,0] = torch.log10(S0[:,0])
S = (S0-S0.mean(0))/S0.std(0)

import args

# init model and optimizer

torch.manual_seed(1)
model = VBGAE3(adj_norm,species_index,2)
init_parameters(model)

optimizer = Adam(model.parameters(), lr=args.learning_rate)

# train model
roclist = []
loss_list= []


torch.manual_seed(1)
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
    
    roclist.append(val_roc2)
    loss_list.append(loss.item())

    pbar.set_postfix({"train_loss=": "{:.5f}".format(loss.item()),
                      'val_roc=': val_roc,
                      "val_roc2=": "{:.5f}".format(val_roc2)})


plt.plot(roclist)
plt.plot(loss_list)

#torch.save(model.state_dict(),"spipoll_results/model")
#model.load_state_dict(torch.load("model",map_location=torch.device("cpu")))


test_roc, test_ap = get_scores(test_edges, test_edges_false, A_pred)
print("1) End of training!", "test_roc=", "{:.5f}".format(test_roc),
      "test_ap=", "{:.5f}".format(test_ap))

test_roc2, test_ap2 = get_scores(test_edges2, test_edges_false2, A_pred2)
print("2) End of training!", "test_roc=", "{:.5f}".format(test_roc2),
      "test_ap=", "{:.5f}".format(test_ap2))


SP = (species01/species01.sum(0)).T.to_numpy()
A_pred3 = (SP@A_pred.detach().numpy())
test_roc3, test_ap3= get_scores(test_edges2, test_edges_false2,torch.Tensor(A_pred3))
print("3) End of training!", "test_roc=", "{:.5f}".format(test_roc3),
      "test_ap=", "{:.5f}".format(test_ap3))

np.round(np.corrcoef(model.mean1.detach().numpy().T,S.T),3)
stat1 = HSIC_stat(model.mean1,S)

x = np.linspace(0, 0.4, 100)
y = stats.gamma.cdf(x,stat1[3].item(),scale=stat1[4].item())
plt.plot(x,y)
plt.axvline(x = stat1[0].item()*n)
stats.gamma.sf(stat1[0].item()*n, stat1[3].item(), scale=stat1[4].item())

####
#torch.save(spipoll_results/model.state_dict(),"model")
#np.savetxt("spipoll_results/A_pred3.csv",A_pred3,delimiter=";")

###### Define model2
torch.manual_seed(2)
model2 = VBGAE3(adj_norm,species_index,2)
init_parameters(model2)

optimizer = Adam(model2.parameters(), lr=args.learning_rate)

# train model2
roclist = []
loss_list= []


torch.manual_seed(2)
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
    
    roclist.append(val_roc2)
    loss_list.append(loss.item())

    pbar.set_postfix({"train_loss=": "{:.5f}".format(loss.item()),
                      'val_roc=': val_roc,
                      "val_roc2=": "{:.5f}".format(val_roc2) ,
                     "HSIC=": "{:.5f}".format(independance)})

#torch.save(model2.state_dict(),"spipoll_results/model2")
#model2.load_state_dict(torch.load("spipoll_results/model2",map_location=torch.device("cpu")))


plt.plot(roclist)
plt.plot(loss_list)


test_roc, test_ap = get_scores(test_edges, test_edges_false, A_pred)
print("1) End of training!", "test_roc=", "{:.5f}".format(test_roc),
      "test_ap=", "{:.5f}".format(test_ap))

test_roc2, test_ap2 = get_scores(test_edges2, test_edges_false2, A_pred2)
print("2) End of training!", "test_roc=", "{:.5f}".format(test_roc2),
      "test_ap=", "{:.5f}".format(test_ap2))


SP = (species01/species01.sum(0)).T.to_numpy()
fair_A_pred3 = (SP@A_pred.detach().numpy())
test_roc3, test_ap3= get_scores(test_edges2, test_edges_false2,torch.Tensor(fair_A_pred3))
print("3) End of training!", "test_roc=", "{:.5f}".format(test_roc3),
      "test_ap=", "{:.5f}".format(test_ap3))

torch.manual_seed(0)
A_pred,A_pred2,Z1,Z2,Z3 = model2(features1,features2)
print(Z1)
print(np.round(np.corrcoef(model2.mean1.detach().numpy().T,S.T),3))

stat2 = HSIC_stat(model2.mean1,S)

x = np.linspace(0, 0.4, 100)
y = stats.gamma.cdf(x,stat2[3].item(),scale=stat2[4].item())
plt.plot(x,y)
plt.axvline(x = stat2[0].item()*n)
stats.gamma.sf(stat2[0].item()*n, stat2[3].item(), scale=stat2[4].item())

#######################################

#np.savetxt("spipoll_results/fair_A_pred3.csv",fair_A_pred3,delimiter=";")

#%%


result = pandas.DataFrame(columns = ["AUC","AP",
                                     "AUC2","AP2",
                                     "AUC3","AP3",
                                     "HSIC","p_value","#0.05","cor2"],
                          index = range(10))



fair_result = pandas.DataFrame(columns = ["AUC","AP",
                                     "AUC2","AP2",
                                     "AUC3","AP3",
                                     "HSIC","p_value","#0.05","cor2"],
                          index = range(10))

torch.manual_seed(1)

n1 = adj.shape[0]

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
    
    
    list_model =  [VBGAE3(adj_norm,species_index,2) for k in range(10)]
    list_val_roc = []
    for model in list_model:
    
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
        list_val_roc.append(val_roc)
        
    best_model = list_model[np.argmax(list_val_roc)]
    latent_space1=best_model.mean1
    latent_space2=best_model.mean2
    A_pred,A_pred2,Z1,Z2,Z3 = best_model(features1,features2)
    A_pred3 = (SP@A_pred.detach().numpy())


    test_roc, test_ap = get_scores(test_edges, test_edges_false, A_pred)
    test_roc2, test_ap2 = get_scores(test_edges2, test_edges_false2, A_pred2)
    test_roc3, test_ap3= get_scores(test_edges2, test_edges_false2,torch.Tensor(A_pred3))
   
    
    cor2 = torch.linalg.norm(torch.corrcoef(torch.cat([latent_space1,S],axis=1).T)[-1,:-1]).item()
    stat1 = HSIC_stat(best_model.mean1,S)
    p005=stats.gamma.sf(stat1[0].item()*n1, stat1[3].item(), scale=stat1[4].item())
    
    result.iloc[k] = [test_roc,test_ap,
                      test_roc2, test_ap2,
                      test_roc3, test_ap3,
                      stat1[0].detach().numpy() ,p005,p005<0.05,cor2]
    print(result.iloc[k])
    result.to_csv("spipoll_results/spipoll_result.csv")
    
    list_model2 =  [VBGAE3(adj_norm,species_index,2) for k in range(10)]
    list_val_roc2 = []
    
    for model2 in list_model2:
    
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
        list_val_roc2.append(val_roc)
        
    best_model2 = list_model2[np.argmax(list_val_roc2)]
    latent_space1=best_model2.mean1
    latent_space2=best_model2.mean2
    A_pred,A_pred2,Z1,Z2,Z3 = best_model2(features1,features2)
    A_pred3 = (SP@A_pred.detach().numpy())


    test_roc, test_ap = get_scores(test_edges, test_edges_false, A_pred)
    test_roc2, test_ap2 = get_scores(test_edges2, test_edges_false2, A_pred2)
    test_roc3, test_ap3= get_scores(test_edges2, test_edges_false2,torch.Tensor(A_pred3))
   
        
    cor2 = torch.linalg.norm(torch.corrcoef(torch.cat([latent_space1,S],axis=1).T)[-1,:-1]).item()
    stat1 = HSIC_stat(best_model2.mean1,S)
    p005=stats.gamma.sf(stat1[0].item()*n1, stat1[3].item(), scale=stat1[4].item())
    
    fair_result.iloc[k] = [test_roc,test_ap,
                      test_roc2, test_ap2,
                      test_roc3, test_ap3,
                      stat1[0].detach().numpy() ,p005,p005<0.05,cor2]
    print(fair_result.iloc[k])
    fair_result.to_csv("spipoll_results/spipoll_fair_result.csv")



print(result)

##############
#%%



adj0=pandas.read_csv("data/net.csv",header=0,sep="\t").to_numpy(dtype=float)
features01 = pandas.read_csv("data/features.csv",header=0,sep="\t")
species01 = pandas.read_csv("data/species.csv",header=0,sep="\t")

mean_Temperature,std_Temperature = features01["Temperature"].mean(),features01["Temperature"].std()

features1 = species01.copy()
features1["Temperature"] = (features01["Temperature"]-mean_Temperature)/std_Temperature


features02 = np.eye(adj0.shape[1])

features1  = torch.Tensor(features1.values)
features2 =  torch.Tensor(features02)

adj_train, train_edges, val_edges, val_edges_false, test_edges, test_edges_false = mask_test_edges(adj0)
adj_train2=torch.Tensor(adj_train.todense())



n=adj0.shape[0]
# Create Model
pos_weight = float(adj0.shape[0] * adj0.shape[1] - adj0.sum()) / adj0.sum()
norm = adj0.shape[0] * adj0.shape[1] / float((adj0.shape[0] * adj0.shape[1] - adj0.sum()) * 2)


adj_label = adj_train 
adj_label = sparse_to_tuple(adj_label)
  
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


SP = (species01/species01.sum(0)).T.to_numpy()
SP = torch.Tensor(SP)
model =  BGAT(features1.shape[1],features2.shape[1],4,4,0.2,10)
optimizer = Adam(model.parameters(), lr=args.learning_rate)
roclist = []
loss_list= []

torch.manual_seed(4)
pbar = tqdm(range(1000),desc = "Training Epochs")
for epoch in pbar:
    t = time.time()

    Z1,Z2 = model(features1,features2,adj_train2)
    A_pred=distance_decode(Z1,Z2)
    A_pred2 = SP@A_pred
    
    optimizer.zero_grad()
    loss = norm*F.binary_cross_entropy(A_pred.view(-1), adj_label.to_dense().view(-1), weight = weight_tensor)
    loss  += norm2*F.binary_cross_entropy(A_pred2.view(-1), torch.Tensor(bipartite).view(-1),weight = weight_tensor2)

    loss.backward()
    optimizer.step()
    

    val_roc, val_ap = get_scores(val_edges, val_edges_false, A_pred)
    val_roc2, val_ap2 = get_scores(val_edges2, val_edges_false2, A_pred2)

    roclist.append(val_roc)
    loss_list.append(loss.item())

    pbar.set_postfix({"train_loss=": "{:.5f}".format(loss.item()),
                                 'val_roc=': val_roc,
                                 "val_roc2=": "{:.5f}".format(val_roc2)})






model2 =  BGAT(features1.shape[1],features2.shape[1],4,4,0.2,10)
optimizer = Adam(model2.parameters(), lr=args.learning_rate)
roclist = []
loss_list= []

torch.manual_seed(4)
pbar = tqdm(range(1000),desc = "Training Epochs")
for epoch in pbar:
    t = time.time()

    Z1,Z2 = model2(features1,features2,adj_train2)
    A_pred=distance_decode(Z1,Z2)
    A_pred2 = SP@A_pred
    
    optimizer.zero_grad()
    loss = norm*F.binary_cross_entropy(A_pred.view(-1), adj_label.to_dense().view(-1), weight = weight_tensor)
    loss  += norm2*F.binary_cross_entropy(A_pred2.view(-1), torch.Tensor(bipartite).view(-1),weight = weight_tensor2)
    independance = n*RFF_HSIC(Z1,S)
    loss += independance
    loss.backward()
    optimizer.step()
    

    val_roc, val_ap = get_scores(val_edges, val_edges_false, A_pred)
    val_roc2, val_ap2 = get_scores(val_edges2, val_edges_false2, A_pred2)

    roclist.append(val_roc)
    loss_list.append(loss.item())

    pbar.set_postfix({"train_loss=": "{:.5f}".format(loss.item()),
                                 'val_roc=': val_roc,
                                 "val_roc2=": "{:.5f}".format(val_roc2)})


stat1 = HSIC_stat(Z1,S)
p005=stats.gamma.sf(stat1[0].item()*n1, stat1[3].item(), scale=stat1[4].item())










