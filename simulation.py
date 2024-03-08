#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on May 2023
Original code by Daehan Kim
GitHub Repository https://github.com/DaehanKim/vgae_pytorch/blob/master/LICENSE

Modified by : ANONYMOUS AUTHOR

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
import matplotlib.patches as mpatches
from preprocessing import *
from model import *
from HSIC import *
import networkx as nx

#os.environ['CUDA_VISIBLE_DEVICES'] = ""

import pandas


n1=1000
n2=100
np.random.seed(1)
z = np.random.normal(size=(n1,1))
s = np.random.normal(size=(n1,1))
#s = np.random.binomial(1, 0.5,size = (n1,1))*2-1

z1 = np.concatenate([s,z],axis=1)
#z1 = np.concatenate([z,s],axis=1)

z2 =  np.random.normal(loc = 2.5,size=(n2,2))

torch.manual_seed(0)
adj0=torch.bernoulli(distance_decode(torch.Tensor(z1),torch.Tensor(z2)))
adj = sp.csr_matrix(adj0) 



G=nx.algorithms.bipartite.from_biadjacency_matrix(adj)
position = {k: np.vstack([z1,z2])[k] for k in G.nodes.keys()}

fig, ax = plt.subplots()
nx.draw_networkx(G,
                 position,
                 node_size=4,
                 with_labels=False,
                 node_color = n1*["#1f77b4"]+n2*["red"],
                 edge_color = (0.75,0.75,0.75),
                 ax=ax)
plt.title("Simulated latent space")
ax.tick_params(left=True, bottom=True, labelleft=True, labelbottom=True)
blue_patch = mpatches.Patch( label=r'$Z_1$')
red_patch = mpatches.Patch(color='red', label=r'$Z_2$')
fig.legend(handles=[blue_patch,red_patch])


plt.scatter(z,s,s=3)
plt.scatter(z2[:,0],z2[:,1],s=3)
plt.title("Simulated latent space")


features01 = np.eye(adj0.shape[0])
features02 = np.eye(adj0.shape[1])

features1 = sp.csr_matrix(features01) 
features2 = sp.csr_matrix(features02) 


features1 = sparse_to_tuple(features1.tocoo())
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


weight_mask = adj_label.to_dense().view(-1) == 1
weight_tensor = torch.ones(weight_mask.size(0)) 
weight_tensor[weight_mask] = pos_weight


##########################################

import args

# init model and optimizer
#2 et 4 
torch.manual_seed(2)
model = VBGAE(adj_norm)
init_parameters(model)

optimizer = Adam(model.parameters(), lr=args.learning_rate)

# train model
roclist = []
loss_list= []


torch.manual_seed(4)
pbar = tqdm(range(1000),desc = "Training Epochs")
for epoch in pbar:
    t = time.time()

    A_pred,Z1,Z2 = model(features1,features2)
    optimizer.zero_grad()
    loss = norm*F.binary_cross_entropy(A_pred.view(-1), adj_label.to_dense().view(-1), weight = weight_tensor)

    kl_divergence = 0.5/ A_pred.size(0) *( (1 + 2*model.logstd1 - model.mean1**2 - torch.exp(model.logstd1)**2).sum(1).mean()+
                                          (1 + 2*model.logstd2 - model.mean2**2 - torch.exp(model.logstd2)**2).sum(1).mean())
    loss -= kl_divergence
    loss.backward()
    optimizer.step()
    

    val_roc, val_ap = get_scores(val_edges, val_edges_false, A_pred)
    
    roclist.append(val_roc)
    loss_list.append(loss.item())

    pbar.set_postfix({"train_loss=": "{:.5f}".format(loss.item()),
                      'val_roc=': val_roc})


plt.plot(roclist)
plt.plot(loss_list)

test_roc, test_ap = get_scores(test_edges, test_edges_false, A_pred)
print("1) End of training!", "test_roc=", "{:.5f}".format(test_roc),
      "test_ap=", "{:.5f}".format(test_ap))

latent_space1=model.mean1
latent_space2=model.mean2



fig = plt.figure()
ax = fig.add_subplot()
plt.scatter(latent_space1[:,0].detach().numpy(),
            latent_space1[:,1].detach().numpy(),
            label=r"$Z_1$",s=20)
plt.scatter(latent_space2[:,0].detach().numpy(),
            latent_space2[:,1].detach().numpy(),
            s=20,label=r"$Z_2$",
            marker="^",c="r")
ax.legend()
plt.title("fair-BVGAE latent space")
plt.show()



fig = plt.figure()
ax = fig.add_subplot()
plt.scatter(latent_space1[:,0].detach().numpy(),
            latent_space1[:,1].detach().numpy(),
            s=10,
            label=r"$Z_1$",
            c=z)
plt.title(r"$\tilde{Z}_1$")
cbar=plt.colorbar(label = "W")
cbar.set_label("W",rotation=0)
plt.show()

fig = plt.figure()
ax = fig.add_subplot()
plt.scatter(latent_space1[:,0].detach().numpy(),
            latent_space1[:,1].detach().numpy(),
            s=10,
            label=r"$Z_1$",
            c=s)
plt.title(r"$\tilde{Z}_1$")
cbar=plt.colorbar(label = "S")
cbar.set_label("S",rotation=0)
plt.show()


fig, axs = plt.subplots(3,figsize=(5,15))

axs[0].scatter(latent_space1[:,0].detach().numpy(),
            latent_space1[:,1].detach().numpy(),
            label=r"$Z_1$",s=20)
axs[0].scatter(latent_space2[:,0].detach().numpy(),
            latent_space2[:,1].detach().numpy(),
            s=20,label=r"$Z_2$",
            marker="^",c="r")
axs[0].legend()
axs[0].title.set_text("BVGAE latent space")

m=axs[1].scatter(latent_space1[:,0].detach().numpy(),
            latent_space1[:,1].detach().numpy(),
            s=20,
            label=r"$Z_1$",
            c=z)
axs[1].title.set_text(r"$\tilde{Z}_1$ colored by $W$")
cbarW=fig.colorbar(m,ax = axs[1],label="W")
cbarW.ax.set_ylabel("W", rotation=0)
nS=axs[2].scatter(latent_space1[:,0].detach().numpy(),
            latent_space1[:,1].detach().numpy(),
            s=20,
            label=r"$Z_1$",
            c=s)
axs[2].title.set_text(r"$\tilde{Z}_1$ colored by $S$")
cbarS=fig.colorbar(nS,ax = axs[2],label="S")
cbarS.ax.set_ylabel('S', rotation=0)


#####


fig = plt.figure()
ax = fig.add_subplot()
plt.scatter(latent_space1[:,0].detach().numpy(),
            latent_space1[:,1].detach().numpy(),
            s=10,
            label=r"$Z_1$",
            c=s)
plt.scatter(latent_space2[:,0].detach().numpy(),
            latent_space2[:,1].detach().numpy(),
            s=20,label=r"$Z_2$",
            marker="^",c="r")
ax.legend()
cbar=plt.colorbar(label = "W")
cbar.set_label("S",rotation=0)
plt.title("BVGAE latent space")
plt.show()

###### plot for the binary network

fig = plt.figure()
ax = fig.add_subplot()
plt.scatter(latent_space1[(s==1).reshape(-1),0].detach().numpy(),
            latent_space1[(s==1).reshape(-1),1].detach().numpy(),
            s=10,
            label=r"$Z_1=1$",
            c="yellow")
plt.scatter(latent_space1[(s==-1).reshape(-1),0].detach().numpy(),
            latent_space1[(s==-1).reshape(-1),1].detach().numpy(),
            s=10,
            label=r"$Z_1=-1$",
            c="purple")

plt.scatter(latent_space2[:,0].detach().numpy(),
            latent_space2[:,1].detach().numpy(),
            s=20,label=r"$Z_2$",
            marker="^",c="r")
ax.legend()
plt.title("BVGAE latent space")
plt.show()



#####


torch.corrcoef(torch.cat([latent_space1,torch.Tensor(z1)],axis=1).T)
S = torch.Tensor(s)

stat2 = HSIC_stat(latent_space1,S)

x = np.linspace(0, 0.4, 100)
y = stats.gamma.cdf(x,stat2[3].item(),scale=stat2[4].item())
plt.plot(x,y)
plt.axvline(x = stat2[0].item()*n)
stats.gamma.sf(stat2[0].item()*n, stat2[3].item(), scale=stat2[4].item())


adj_fair = (A_pred>0.5).numpy()*1
G=nx.algorithms.bipartite.from_biadjacency_matrix( sp.csr_matrix(adj_fair))
position = {k: np.vstack([latent_space1.detach().numpy(),latent_space2.detach().numpy()])[k] for k in G.nodes.keys()}
fig, ax = plt.subplots()
nx.draw_networkx(G,
                 position,
                 node_size=4,
                 with_labels=False,
                 node_color = n1*["#1f77b4"]+n2*["red"],
                 edge_color = (0.75,0.75,0.75),
                 ax=ax)
plt.title("Simulated latent space")
ax.tick_params(left=True, bottom=True, labelleft=True, labelbottom=True)



##fair model


torch.manual_seed(5)#16
model2 = VBGAE(adj_norm)
init_parameters(model)

optimizer = Adam(model2.parameters(), lr=args.learning_rate)

# train model
roclist = []
loss_list= []
indep_list = []

torch.manual_seed(3)#3
pbar = tqdm(range(1000),desc = "Training Epochs")
for epoch in pbar:
    t = time.time()

    A_pred,Z1,Z2 = model2(features1,features2)
    optimizer.zero_grad()
    loss = norm*F.binary_cross_entropy(A_pred.view(-1), adj_label.to_dense().view(-1), weight = weight_tensor)

    kl_divergence = 0.5/ A_pred.size(0) *( (1 + 2*model2.logstd1 - model2.mean1**2 - torch.exp(model2.logstd1)**2).sum(1).mean()+
                                          (1 + 2*model2.logstd2 - model2.mean2**2 - torch.exp(model2.logstd2)**2).sum(1).mean())
    loss -= kl_divergence
    independance =2000*(RFF_HSIC(model2.mean1,S))#0.02
    #independance =2*torch.log(RFF_HSIC(Z1,S))#0.02

    loss += independance
    loss.backward()
    optimizer.step()
    

    val_roc, val_ap = get_scores(val_edges, val_edges_false, A_pred)
    
    roclist.append(val_roc)
    loss_list.append(loss.item())
    indep_list.append(independance.item())

    pbar.set_postfix({"train_loss=": "{:.5f}".format(loss.item()),
                      'val_roc=': val_roc,'log_HSIC =': independance.item()})


test_roc, test_ap = get_scores(test_edges, test_edges_false, A_pred)
print("1) End of training!", "test_roc=", "{:.5f}".format(test_roc),
      "test_ap=", "{:.5f}".format(test_ap))


plt.plot(roclist)
plt.plot(loss_list)
plt.plot(indep_list)



latent_space1=model2.mean1
latent_space2=model2.mean2



torch.corrcoef(torch.cat([latent_space1,torch.Tensor(z1)],axis=1).T)

stat2 = HSIC_stat(latent_space1,S)

x = np.linspace(0, 0.4, 100)
y = stats.gamma.cdf(x,stat2[3].item(),scale=stat2[4].item())
plt.plot(x,y)
plt.axvline(x = stat2[0].item()*n)
stats.gamma.sf(stat2[0].item()*n, stat2[3].item(), scale=stat2[4].item())



fig = plt.figure()
ax = fig.add_subplot()
plt.scatter(latent_space1[:,0].detach().numpy(),
            latent_space1[:,1].detach().numpy(),
            label="Collection",s=1)
plt.scatter(latent_space2[:,0].detach().numpy(),
            latent_space2[:,1].detach().numpy(),
            s=5,label="Observed insects",
            marker="^",c="r")
ax.legend( bbox_to_anchor=(1, -0.1))
plt.title("2D latent space")
plt.show()



fig = plt.figure()
ax = fig.add_subplot()
plt.scatter(latent_space1[:,0].detach().numpy(),
            latent_space1[:,1].detach().numpy(),
            label="Collection",c=z,s=15)
ax.legend( bbox_to_anchor=(1, -0.1))
plt.title("2D latent space")
plt.colorbar(label = "Z")
plt.show()

fig = plt.figure()
ax = fig.add_subplot()
plt.scatter(latent_space1[:,0].detach().numpy(),
            latent_space1[:,1].detach().numpy(),
            label="Collection",c=s)
ax.legend( bbox_to_anchor=(1, -0.1))
plt.title("2D latent space")
plt.colorbar(label = "S")
plt.show()



fig, axs = plt.subplots(3,figsize=(5,15))

axs[0].scatter(latent_space1[:,0].detach().numpy(),
            latent_space1[:,1].detach().numpy(),
            label=r"$Z_1$",s=20)
axs[0].scatter(latent_space2[:,0].detach().numpy(),
            latent_space2[:,1].detach().numpy(),
            s=20,label=r"$Z_2$",
            marker="^",c="r")
axs[0].legend()
axs[0].title.set_text("fair-BVGAE latent space")

m=axs[1].scatter(latent_space1[:,0].detach().numpy(),
            latent_space1[:,1].detach().numpy(),
            s=20,
            label=r"$Z_1$",
            c=z)
axs[1].title.set_text(r"fair-$\tilde{Z}_1$ colored by $W$")
cbarW=fig.colorbar(m,ax = axs[1],label="W")
cbarW.ax.set_ylabel("W", rotation=0)
nS=axs[2].scatter(latent_space1[:,0].detach().numpy(),
            latent_space1[:,1].detach().numpy(),
            s=20,
            label=r"$Z_1$",
            c=s)
axs[2].title.set_text(r"fair-$\tilde{Z}_1$ colored by $S$")
cbarS=fig.colorbar(nS,ax = axs[2],label="S")
cbarS.ax.set_ylabel('S', rotation=0)


#####


fig = plt.figure()
ax = fig.add_subplot()
plt.scatter(latent_space2[:,0].detach().numpy(),
            latent_space2[:,1].detach().numpy(),
            s=20,label=r"$Z_2$",
            marker="^",c="r")
plt.scatter(latent_space1[:,0].detach().numpy(),
            latent_space1[:,1].detach().numpy(),
            s=1,
            label=r"$Z_1$",
            c=s)

ax.legend()
cbar=plt.colorbar(label = "W")
cbar.set_label("S",rotation=0)
plt.title("fair-BVGAE latent space")
plt.show()

#####plot for binary network
fig = plt.figure()
ax = fig.add_subplot()
plt.scatter(latent_space1[(s==1).reshape(-1),0].detach().numpy(),
            latent_space1[(s==1).reshape(-1),1].detach().numpy(),
            s=10,
            label=r"$Z_1=1$",
            c="yellow")
plt.scatter(latent_space1[(s==-1).reshape(-1),0].detach().numpy(),
            latent_space1[(s==-1).reshape(-1),1].detach().numpy(),
            s=10,
            label=r"$Z_1=-1$",
            c="purple")

plt.scatter(latent_space2[:,0].detach().numpy(),
            latent_space2[:,1].detach().numpy(),
            s=20,label=r"$Z_2$",
            marker="^",c="r")
ax.legend()
plt.title("fair-BVGAE latent space")
plt.show()




#####


A_pred=distance_decode(latent_space1,latent_space2)
adj_fair = (A_pred>0.5).numpy()*1
G=nx.algorithms.bipartite.from_biadjacency_matrix( sp.csr_matrix(adj_fair))
position = {k: np.vstack([latent_space1.detach().numpy(),latent_space2.detach().numpy()])[k] for k in G.nodes.keys()}
fig, ax = plt.subplots()
nx.draw_networkx(G,
                 position,
                 node_size=4,
                 with_labels=False,
                 node_color = n1*["#1f77b4"]+n2*["red"],
                 edge_color = (0.75,0.75,0.75),
                 ax=ax)
plt.title("Simulated latent space")
ax.tick_params(left=True, bottom=True, labelleft=True, labelbottom=True)


#########################################


result = pandas.DataFrame(columns = ["AUC","AP","HSIC","p_value","#0.05","cov_1"],
                          index = range(100))



for k in range(100):
    model = VBGAE(adj_norm)
    init_parameters(model)
    optimizer = Adam(model.parameters(), lr=args.learning_rate*2)
    
    # train model
    pbar = tqdm(range(1000),desc = "Training Epochs")
    for epoch in pbar:
        t = time.time()
    
        A_pred,Z1,Z2 = model(features1,features2)
        optimizer.zero_grad()
        loss = norm*F.binary_cross_entropy(A_pred.view(-1), adj_label.to_dense().view(-1), weight = weight_tensor)
        kl_divergence = 0.5/ A_pred.size(0) *( (1 + 2*model.logstd1 - model.mean1**2 - torch.exp(model.logstd1)**2).sum(1).mean()+
                                              (1 + 2*model.logstd2 - model.mean2**2 - torch.exp(model.logstd2)**2).sum(1).mean())
        loss -= kl_divergence
        loss.backward()
        optimizer.step()
        
        val_roc, val_ap = get_scores(val_edges, val_edges_false, A_pred)
        pbar.set_postfix({"train_loss=": "{:.5f}".format(loss.item()),
                          'val_roc=': val_roc})
    
     
    latent_space1=model.mean1
    latent_space2=model.mean2
    A_pred=distance_decode(latent_space1,latent_space2)
    test_roc, test_ap = get_scores(test_edges, test_edges_false, A_pred)
    torch.corrcoef(torch.cat([latent_space1,torch.Tensor(z1)],axis=1).T)
    stat2 = HSIC_stat(latent_space1,S)
    zut=stats.gamma.sf(stat2[0].item()*n, stat2[3].item(), scale=stat2[4].item())
    cov2 = torch.linalg.norm(torch.corrcoef(torch.cat([latent_space1,torch.Tensor(z1)],axis=1).T)[2,:2]).item()

    result.iloc[k] = [test_roc,test_ap,stat2[0].detach().numpy() ,zut,zut<0.05,cov2]
    print(result.iloc[k])








fair_result = pandas.DataFrame(columns = ["AUC","AP","HSIC","p_value","#0.05","cov_1"],
                          index = range(100))



for k in range(100):
    model2 = VBGAE(adj_norm)
    init_parameters(model)
    
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
        independance =1000*(RFF_HSIC(model2.mean1,S))#0.02
    
        loss += independance
        loss.backward()
        optimizer.step()
        
    
        val_roc, val_ap = get_scores(val_edges, val_edges_false, A_pred)
        pbar.set_postfix({"train_loss=": "{:.5f}".format(loss.item()),
                          'val_roc=': val_roc,'log_HSIC =': independance.item()})
    
    
    test_roc, test_ap = get_scores(test_edges, test_edges_false, A_pred)

    latent_space1=model2.mean1
    latent_space2=model2.mean2
    torch.corrcoef(torch.cat([latent_space1,torch.Tensor(z1)],axis=1).T)
    stat2 = HSIC_stat(latent_space1,S)
    zut=stats.gamma.sf(stat2[0].item()*n, stat2[3].item(), scale=stat2[4].item())
    cov2 = torch.linalg.norm(torch.corrcoef(torch.cat([latent_space1,torch.Tensor(z1)],axis=1).T)[2,:2]).item()

    fair_result.iloc[k] = [test_roc,test_ap,stat2[0].detach().numpy() ,zut,zut<0.05,cov2]
    print(fair_result.iloc[k])


##########################



result = pandas.DataFrame(columns = ["AUC","AP","HSIC","p_value","#0.05","cov_1"],
                          index = range(100))



fair_result = pandas.DataFrame(columns = ["AUC","AP","HSIC","p_value","#0.05","cov_1"],
                          index = range(100))


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

for k in range(100):
    print(10*"#")
    print(k)
    print(10*"#")
    
    z = np.random.normal(size=(n1,1))
    #s = np.random.normal(size=(n1,1))
    s = np.random.binomial(1, 0.5,size = (n1,1))*2-1
    S = torch.Tensor(s)
    z1 = np.concatenate([s,z],axis=1)
    z2 =  np.random.normal(loc = 2.5,size=(n2,2))
    

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

    
    list_model =  [VBGAE(adj_norm) for k in range(10)]
    list_val_roc = []
    for model in list_model:
        init_parameters(model)
        optimizer = Adam(model.parameters(), lr=args.learning_rate*2)
        
        # train model
        pbar = tqdm(range(1000),desc = "Training Epochs")
        for epoch in pbar:
            t = time.time()
        
            A_pred,Z1,Z2 = model(features1,features2)
            optimizer.zero_grad()
            loss = norm*F.binary_cross_entropy(A_pred.view(-1), adj_label.to_dense().view(-1), weight = weight_tensor)
            kl_divergence = 0.5/ A_pred.size(0) *( (1 + 2*model.logstd1 - model.mean1**2 - torch.exp(model.logstd1)**2).sum(1).mean()+
                                                  (1 + 2*model.logstd2 - model.mean2**2 - torch.exp(model.logstd2)**2).sum(1).mean())
            loss -= kl_divergence
            loss.backward()
            optimizer.step()
            
            val_roc, val_ap = get_scores(val_edges, val_edges_false, A_pred)
            pbar.set_postfix({"train_loss=": "{:.5f}".format(loss.item()),
                              'val_roc=': val_roc})
        list_val_roc.append(val_roc)
        
        
    best_model = list_model[np.argmax(list_val_roc)]
    latent_space1=best_model.mean1
    latent_space2=best_model.mean2
    A_pred,Z1,Z2 = best_model(features1,features2)
    test_roc, test_ap = get_scores(test_edges, test_edges_false, A_pred)
    stat2 = HSIC_stat(latent_space1,S)
    zut=stats.gamma.sf(stat2[0].item()*n1, stat2[3].item(), scale=stat2[4].item())
    cov2 = torch.linalg.norm(torch.corrcoef(torch.cat([latent_space1,torch.Tensor(z1)],axis=1).T)[2,:2]).item()

    result.iloc[k] = [test_roc,test_ap,stat2[0].detach().numpy() ,zut,zut<0.05,cov2]
    print(result.iloc[k])


    list_model2 =  [VBGAE(adj_norm) for k in range(10)]
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
            independance =n1*(RFF_HSIC(model2.mean1,S))#0.02
        
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

    fair_result.iloc[k] = [test_roc,test_ap,stat2[0].detach().numpy() ,zut,zut<0.05,cov2]
    print(fair_result.iloc[k])
    result.to_csv("binary_result2.csv")
    fair_result.to_csv("binary_fair_result2.csv")







adv_result = pandas.DataFrame(columns = ["AUC","AP","HSIC","p_value","#0.05","cov_1"],
                          index = range(100))




for k in range(100):
    print(10*"#")
    print(k)
    print(10*"#")
    
    z = np.random.normal(size=(n1,1))
    s = np.random.normal(size=(n1,1))
    S = torch.Tensor(s)
    z1 = np.concatenate([s,z],axis=1)
    z2 =  np.random.normal(loc = 2.5,size=(n2,2))
    
    
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
    
    
    
    
    
    
    
    
    list_model3 =  [VBGAE(adj_norm) for k in range(10)]
    list_adv3 =  [Adversary(1) for k in range(10)]
    list_val_roc3 = []
    for model,adv in zip(list_model3,list_adv3):
        #init_parameters(model)
        init_parameters(adv)
        optimizer = Adam(model.parameters(), lr=args.learning_rate*2)
        adv_optimizer = Adam(adv.parameters(),lr = 0.01)
    
        # train model
        pbar = tqdm(range(400),desc = "Training GVAE")
        for epoch in pbar:
            t = time.time()
        
            A_pred,Z1,Z2 = model(features1,features2)
            optimizer.zero_grad()
            loss = norm*F.binary_cross_entropy(A_pred.view(-1), adj_label.to_dense().view(-1), weight = weight_tensor)
            kl_divergence = 0.5/ A_pred.size(0) *( (1 + 2*model.logstd1 - model.mean1**2 - torch.exp(model.logstd1)**2).sum(1).mean()+
                                                  (1 + 2*model.logstd2 - model.mean2**2 - torch.exp(model.logstd2)**2).sum(1).mean())
            loss -= kl_divergence
            loss.backward()
            optimizer.step()
            
            val_roc, val_ap = get_scores(val_edges, val_edges_false, A_pred)
            pbar.set_postfix({"train_loss=": "{:.5f}".format(loss.item()),
                              'val_roc=': val_roc})
        
        pbar = tqdm(range(1000),desc = "Training adversary")
        for epoch in pbar:
            t = time.time()
            adv_optimizer.zero_grad()
            A_pred,Z1,Z2 = model(features1,features2)
            Z1 = Z1.detach()
            s_hat = adv(Z1)
            adv_loss = -torch.abs(torch.corrcoef(torch.cat([S,s_hat],axis=1).T)[0,1])
            adv_loss.backward()
            adv_optimizer.step()
            
            pbar.set_postfix({"adv_loss=": "{:.5f}".format(adv_loss.item())})
            
            
        
        pbar = tqdm(range(600),desc = "Adversarial training")
        for epoch in pbar:
            #Train adversary
            t = time.time()
            adv_optimizer.zero_grad()
            _,Z1,_ = model(features1,features2)
            Z1 = Z1.detach()
            s_hat = adv(Z1)
            adv_loss = -torch.abs(torch.corrcoef(torch.cat([S,s_hat],axis=1).T)[0,1])
            adv_loss.backward()
            adv_optimizer.step()
            
            #Train GVAE
            A_pred,Z1,Z2 = model(features1,features2)
            optimizer.zero_grad()
            loss = norm*F.binary_cross_entropy(A_pred.view(-1), adj_label.to_dense().view(-1), weight = weight_tensor)
            kl_divergence = 0.5/ A_pred.size(0) *( (1 + 2*model.logstd1 - model.mean1**2 - torch.exp(model.logstd1)**2).sum(1).mean()+
                                                  (1 + 2*model.logstd2 - model.mean2**2 - torch.exp(model.logstd2)**2).sum(1).mean())
            loss -= kl_divergence
            s_hat = adv(Z1)
            adv_loss = -torch.abs(torch.corrcoef(torch.cat([S,s_hat],axis=1).T)[0,1])
            loss -= adv_loss*torch.tensor(epoch)
            loss.backward()
            optimizer.step()
            
            val_roc, val_ap = get_scores(val_edges, val_edges_false, A_pred)
            pbar.set_postfix({"train_loss=": "{:.5f}".format(loss.item()),
                              'val_roc=': val_roc})
            
            
        list_val_roc3.append(val_roc)
        print(10*"#")
    best_model = list_model3[np.argmax(list_val_roc3)]    
    latent_space1=best_model.mean1
    latent_space2=best_model.mean2
    A_pred,Z1,Z2 = best_model(features1,features2)
    test_roc, test_ap = get_scores(test_edges, test_edges_false, A_pred)
    stat2 = HSIC_stat(latent_space1,S)
    zut=stats.gamma.sf(stat2[0].item()*n1, stat2[3].item(), scale=stat2[4].item())
    cov2 = torch.linalg.norm(torch.corrcoef(torch.cat([latent_space1,torch.Tensor(z1)],axis=1).T)[2,:2]).item()
    
    adv_result.iloc[k] = [test_roc,test_ap,stat2[0].detach().numpy() ,zut,zut<0.05,cov2]
    print(adv_result.iloc[k])




delta_hyperparameter_table = pandas.DataFrame(columns = ["AUC","AP","HSIC","p_value","#0.05","cov_1"])






