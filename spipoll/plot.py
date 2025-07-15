
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


import imageio.v2 as imageio




##########


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


species_index =  np.array((np.where(species01))).T[:,1]

bipartite,val_edges2,val_edges_false2,test_edges2,test_edges_false2=mask_test_edges2(adj_label,species01.to_numpy(), val_edges, val_edges_false, test_edges, test_edges_false)

pos_weight2 = (bipartite.shape[0]*bipartite.shape[1]-bipartite.sum())/(bipartite.sum())
weight_tensor2 = torch.ones(bipartite.reshape(-1).shape[0]) 
weight_tensor2[bipartite.reshape(-1)==1] = pos_weight2

norm2 = bipartite.shape[0] * bipartite.shape[1] / float((bipartite.shape[0] *bipartite.shape[1] - bipartite.sum()) * 2)

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
args.hidden2_dim1 = 4
args.hidden2_dim2 = 4


#%%
##### REALISISATION DE L'ESPACE LATENT 1 

model = VBGAE3(adj_norm,species_index,2)
model.load_state_dict(torch.load("spipoll_results/model2",map_location=torch.device("cpu")))
torch.manual_seed(2)
_,_,latent_space1,latent_space2,_ = model(features1,features2)
latent_space1=latent_space1.detach().numpy()
latent_space2=latent_space2.detach().numpy()

#latent_space1 = model2.mean1.detach().numpy()
#latent_space2 = model2.mean2.detach().numpy()




ax0 = 0
ax1 = 1

fig = plt.figure()
ax = fig.add_subplot()

plt.scatter(latent_space1[:,ax0],latent_space1[:,ax1],
            label="Collection",s=1,
            c=np.log10(S0[:,0].numpy()))
plt.scatter(latent_space2[:,ax0],latent_space2[:,ax1],
            s=5,label="Observed insects",
            marker="^",c="r")
#ax.set_aspect('equal', adjustable='box')
ax.legend( bbox_to_anchor=(1, -0.1))
plt.title("2D latent space")
cbar = plt.colorbar()
cbar.set_label('log10 user exp', rotation=270,labelpad=15)
plt.show()



#%%
##### REALISISATION DE L'ESPACE LATENT 1 

model = VBGAE3(adj_norm,species_index,2)
model.load_state_dict(torch.load("spipoll_results/model2",map_location=torch.device("cpu")))
torch.manual_seed(2)
_,_,latent_space1,latent_space2,_ = model(features1,features2)
latent_space1=latent_space1.detach().numpy()
latent_space2=latent_space2.detach().numpy()

fig,axs = plt.subplots(args.hidden2_dim1-1,args.hidden2_dim1-1,figsize = (15,15))

for i in range(args.hidden2_dim1,):
    for j in range(i+1,args.hidden2_dim1,):
        axs[i,j-1].axvline(0)
        axs[i,j-1].axhline(0)
        A=axs[i,j-1].scatter(latent_space1[:,i],latent_space1[:,j],
                    label="Collection",s=1,
                    c=np.log10(S0[:,0].numpy()))
        axs[i,j-1].scatter(latent_space2[:,i],latent_space2[:,j],
                    s=5,label="Observed insects",
                    marker="^",c="r")
plt.setp(axs, xlim=(-6,6), ylim=(-6,6))

for i in range(1,args.hidden2_dim1-1):
    for j in range(i):
        axs[i,j].axis("off")
        
                

for j in range(args.hidden2_dim1-1):
    axs[0,j].set_title("dim "+str(j+2))
        
for i in range(args.hidden2_dim1-1):
    axs[i,i].set_ylabel("dim " +str(i+1))


handles, labels = axs[0,0].get_legend_handles_labels()
legend=axs[1,0].legend(handles, labels, loc='lower center', prop={'size': 20})
for handle in legend.legendHandles:
    handle.set_sizes([60.0])
cb=fig.colorbar(A,ax= axs[args.hidden2_dim1-2,:args.hidden2_dim1-2],location='top')
cb.set_label(label=r"$log_{10}$ user exp",size=20)
cb.ax.xaxis.set_label_position("bottom")
plt.show()


#%%

model = VBGAE3(adj_norm,species_index,2)
model.load_state_dict(torch.load("spipoll_results/model2",map_location=torch.device("cpu")))
torch.manual_seed(2)
_,_,latent_space1,latent_space2,_ = model(features1,features2)
latent_space1=latent_space1.detach().numpy()
latent_space2=latent_space2.detach().numpy()

#latent_space1 = model2.mean1.detach().numpy()
#latent_space2 = model2.mean2.detach().numpy()



plant_genus = species01.iloc[:,5]
plant_genus2 = species01.iloc[:,1]
ax0 = 0
ax1 = 1

fig = plt.figure()
ax = fig.add_subplot()

plt.scatter(latent_space1[:,ax0][plant_genus==1],latent_space1[:,ax1][plant_genus==1],
            label="Trifolium",s=4)
plt.scatter(latent_space1[:,ax0][plant_genus2==1],latent_space1[:,ax1][plant_genus2==1],
            label="Leucanthemum",s=4)
plt.scatter(latent_space2[:,ax0],latent_space2[:,ax1],
            s=2,label="Observed insects",
            marker="^",c="r")
#ax.set_aspect('equal', adjustable='box')
ax.legend( bbox_to_anchor=(1, -0.1))
plt.title("Latent space")
ax.set_box_aspect(1)


#frame0 = 2.5
#plt.setp(ax, xlim=(-frame0,fram0), ylim=(-frame0,frame0))

plt.show()

#%%

model = VBGAE3(adj_norm,species_index,2)
model.load_state_dict(torch.load("spipoll_results/model2",map_location=torch.device("cpu")))
torch.manual_seed(2)
_,_,latent_space1,latent_space2,_ = model(features1,features2)
plant_genus = species01.iloc[:,15]
plant_genus2 = species01.iloc[:,1]
latent_space1=latent_space1.detach().numpy()
latent_space2=latent_space2.detach().numpy()
frame0 = 3
fig,axs = plt.subplots(args.hidden2_dim1-1,args.hidden2_dim1-1,figsize = (15,15))

for i in range(args.hidden2_dim1,):
    for j in range(i+1,args.hidden2_dim1,):
        axs[i,j-1].axvline(0)
        axs[i,j-1].axhline(0)
        A=axs[i,j-1].scatter(latent_space1[:,i][plant_genus==1],latent_space1[:,j][plant_genus==1],
                    label="Daucus",s=4)
        A=axs[i,j-1].scatter(latent_space1[:,i][plant_genus2==1],latent_space1[:,j][plant_genus2==1],
                    label="Leucanthemum",s=4)
        axs[i,j-1].scatter(latent_space2[:,i],latent_space2[:,j],
                    s=2,label="Observed insects",
                    marker="^",c="r",alpha=0.5)
plt.setp(axs,  xlim=(-frame0,frame0), ylim=(-frame0,frame0))

for i in range(1,args.hidden2_dim1-1):
    for j in range(i):
        axs[i,j].axis("off")
        
                

for j in range(args.hidden2_dim1-1):
    axs[0,j].set_title("dim "+str(j+2))
        
for i in range(args.hidden2_dim1-1):
    axs[i,i].set_ylabel("dim " +str(i+1))


handles, labels = axs[0,0].get_legend_handles_labels()
legend=axs[2,0].legend(handles, labels, loc='lower center', prop={'size': 30})
for handle in legend.legend_handles:
    handle.set_sizes([160.0])
#plt.savefig("spipoll_results/Daucus_Leucanthemum_latent_space.pdf")




#%%

model = VBGAE3(adj_norm,species_index,2)
model.load_state_dict(torch.load("spipoll_results/model2",map_location=torch.device("cpu")))
torch.manual_seed(2)
_,_,latent_space1,latent_space2,_ = model(features1,features2)
latent_space1=latent_space1.detach().numpy()
latent_space2=latent_space2.detach().numpy()

plant_genus = species01.iloc[:,35]
ax0 = 0
ax1 = 1

fig = plt.figure()
ax = fig.add_subplot()
ax.axvline(0)
ax.axhline(0)
plt.scatter(latent_space1[:,ax0][plant_genus==1],latent_space1[:,ax1][plant_genus==1],
            label="Trifolium",s=20,c = features01["Temperature"][plant_genus==1])

#ax.set_aspect('equal', adjustable='box')
ax.legend( bbox_to_anchor=(1, -0.1))
plt.title("Latent space")
ax.set_box_aspect(1)


plt.show()



#%%

model = VBGAE3(adj_norm,species_index,2)
model.load_state_dict(torch.load("spipoll_results/model2",map_location=torch.device("cpu")))
torch.manual_seed(2)
_,_,latent_space1,latent_space2,_ = model(features1,features2)
latent_space1=latent_space1.detach().numpy()
latent_space2=latent_space2.detach().numpy()
plant_genus = species01.iloc[:,35]

fig,axs = plt.subplots(args.hidden2_dim1-1,args.hidden2_dim1-1,figsize = (15,15))

for i in range(args.hidden2_dim1,):
    for j in range(i+1,args.hidden2_dim1,):
        axs[i,j-1].axvline(0)
        axs[i,j-1].axhline(0)
        axs[i,j-1].scatter(latent_space2[:,i],latent_space2[:,j],
                    s=2,label="Observed insects",
                    marker="^",c="r")
        A=axs[i,j-1].scatter(latent_space1[:,i][plant_genus==1],latent_space1[:,j][plant_genus==1],
                    label="Senecio",s=4,
                    c=features01["Temperature"][plant_genus==1])

plt.setp(axs, xlim=(-4,4), ylim=(-4,4))

for i in range(1,args.hidden2_dim1-1):
    for j in range(i):
        axs[i,j].axis("off")
        
                

for j in range(args.hidden2_dim1-1):
    axs[0,j].set_title("dim "+str(j+2))
        
for i in range(args.hidden2_dim1-1):
    axs[i,i].set_ylabel("dim " +str(i+1))


handles, labels = axs[0,0].get_legend_handles_labels()
legend=axs[1,0].legend(handles, labels, loc='lower center', prop={'size': 20})
for handle in legend.legend_handles:
    handle.set_sizes([60.0])
cb=fig.colorbar(A,ax= axs[args.hidden2_dim1-2,:args.hidden2_dim1-2],location='top')
cb.set_label(label=r"Temperature (C°)",size=20)
cb.ax.xaxis.set_label_position("bottom")
plt.savefig("spipoll_results/latent_space_temperature.pdf")
plt.show()


#%%

model = VBGAE3(adj_norm,species_index,2)
model.load_state_dict(torch.load("spipoll_results/model2",map_location=torch.device("cpu")))
torch.manual_seed(2)
_,_,latent_space1,latent_space2,_ = model(features1,features2)
plant_genus = species01.iloc[:,15]
plant_genus2 = species01.iloc[:,1]
plant_genus3 = species01.iloc[:,5]

latent_space1=latent_space1.detach().numpy()
latent_space2=latent_space2.detach().numpy()
frame0 = 1.5
fig,axs = plt.subplots(args.hidden2_dim1-1,args.hidden2_dim1-1,figsize = (15,15))

for i in range(args.hidden2_dim1,):
    for j in range(i+1,args.hidden2_dim1,):
        axs[i,j-1].axvline(0)
        axs[i,j-1].axhline(0)
        A=axs[i,j-1].scatter(latent_space1[:,i][plant_genus==1],latent_space1[:,j][plant_genus==1],
                    label="Daucus",s=4,c="C1")
        A=axs[i,j-1].scatter(latent_space1[:,i][plant_genus2==1],latent_space1[:,j][plant_genus2==1],
                    label="Leucanthemum",s=4,c="C0")
        A=axs[i,j-1].scatter(latent_space1[:,i][plant_genus3==1],latent_space1[:,j][plant_genus3==1],
                    label="Rosmarinus",s=4,c="C2")
        axs[i,j-1].scatter(latent_space2[:,i],latent_space2[:,j],
                    s=16,label="Observed insects",
                    marker="^",c="r",alpha=0.5)
plt.setp(axs,  xlim=(-frame0,frame0), ylim=(-frame0,frame0))

for i in range(1,args.hidden2_dim1-1):
    for j in range(i):
        axs[i,j].axis("off")
        
                

    
    
for i in range(args.hidden2_dim1-1):
    for j in range(i,args.hidden2_dim1-1):
        axs[i,j].set_xlabel("dim "+str(i+1))
        axs[i,j].set_ylabel("dim "+str(j+2))


plt.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.9, wspace=0.4,hspace=0.4)


handles, labels = axs[0,0].get_legend_handles_labels()
legend=axs[2,0].legend(handles, labels, loc='lower center', prop={'size': 30})
for handle in legend.legend_handles:
    handle.set_sizes([160.0])
#plt.savefig("spipoll_results/Daucus_Leucanthemum_latent_space.pdf")


