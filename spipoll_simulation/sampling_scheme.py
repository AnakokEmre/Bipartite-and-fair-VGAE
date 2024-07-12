#!/usr/bin/env python
# coding: utf-8

# # Expérience des utilisateurs :

# 1 couleur = 1 utilisateur


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


# In[122]:

def mask_test_edges3(adj_label,species, ground_truth, val_edges, val_edges_false, test_edges, test_edges_false):
        
    bipartite = 1*(species.T.dot(adj_label.to_dense().numpy())>0) #bipartite observé (train)
    forbidden = (bipartite==1) #on  ne test pas sur les arrêtes observées
    val_edges2 = np.zeros(adj_label.shape) 
    val_edges2[val_edges]=1
    val_edges2=species.T.dot(val_edges2)
    val_edges2= (val_edges2>0)& np.logical_not(forbidden) #contient les arrêtes de validation
    forbidden = forbidden + val_edges2 #update forbidden
    
    val_edges_false2=np.zeros(adj_label.shape) 
    val_edges_false2[val_edges_false]=1
    val_edges_false2=species.T.dot(val_edges_false2)
    val_edges_false2=(val_edges_false2>0) & np.logical_not(forbidden)#contient les non arrêtes de validation
    forbidden = forbidden + val_edges_false2
    
    
    #test_edges2 = np.zeros(adj_label.shape)
    #test_edges2[test_edges]=1
    test_edges2 = ground_truth
    test_edges2 = (test_edges2>0)& np.logical_not(forbidden)#contient les arrêtes de test
    forbidden = forbidden + test_edges2
    
    
    #test_edges_false2=np.zeros(adj_label.shape)
    #test_edges_false2[test_edges_false]=1
    
    #test_edges_false2=species.T.dot(test_edges_false2) #contient les non-arrêtes de test
    test_edges_false2 = 1-ground_truth
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




#bipartite_net = np.random.randint(2,size=(83,306))

def simulate_lbm(n1,n2,alpha,beta,P):
    W1 = np.random.choice(len(alpha),replace=True,p=alpha, size=n1)
    W2 = np.random.choice(len(beta) ,replace=True,p=beta , size=n2)
    proba = (P[W1].T[W2]).T
    M = np.random.binomial(1,proba)
    return W1,W2,M

alpha = (0.3,0.7)
beta = (0.3,0.7)
P = np.array([[0.9,0.5],[0.6,0.2]])


alpha = (0.3,0.4,0.3)
beta = (0.2,0.4,0.4)
P = np.array([[0.95,0.80,0.5],
              [0.90,0.55,0.2],
              [0.7,0.25,0.06]])

n01=83
n02=306
W1,W2,bipartite_net = simulate_lbm(n01, n02, alpha, beta, P) 
plt.imshow(1-bipartite_net[np.argsort(W1),:][:,np.argsort(W2)],interpolation="none",cmap = "gray")


# ## EXAMPLE 0 

# In[123]:


W3 = np.ones(len(W1))
#W4 = np.random.binomial(1,np.array([0.1,1])[W2])
W4 = np.random.binomial(1,np.array([0.1,0.4,0.9])[W2])

# In[124]:


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
plt.imshow(1*(bipartite_obs==0)[np.argsort(W1),:][:,np.argsort(W2)],interpolation="none",cmap="gray")


# In[125]:


print((bipartite_net>0).sum())


# In[126]:




# In[127]:


print(((species.T@net0)>0).sum())


# In[128]:




# In[129]:


print(1*(bipartite_obs>0).sum())


# In[130]:




# In[131]:


print((bipartite_obs>0)[W1==0,:][:,W2==0].mean())
print((bipartite_obs>0)[W1==1,:][:,W2==0].mean())
print((bipartite_obs>0)[W1==0,:][:,W2==1].mean())
print((bipartite_obs>0)[W1==1,:][:,W2==1].mean())


# In[ ]:





# In[132]:


plt.imshow((bipartite_obs>0)[np.argsort(W1),:][:,np.argsort(W2)],interpolation="none")



# In[ ]:

truc = np.array([0,0,1,0]).reshape(2,2)
plt.imshow(truc>0,cmap="Greys",interpolation="none")


plt.imshow(net0,cmap="Greys",interpolation="none")


# In[ ]:





# In[134]:


plt.scatter(user_exp,net0.sum(1))


# In[135]:
truc = pandas.DataFrame({"exp":user_exp,
                         "nb_obs":nb_obs,
                         "nb_obs0": net0[:,(W4==0)].sum(1),
                         "nb_obs1": net0[:,(W4==1)].sum(1),})

    
truc_mean=truc.groupby(["exp"]).mean()

plt.figure(figsize=(6,6))
plt.scatter(truc_mean.index,truc_mean["nb_obs0"],c="red",marker="x")
plt.scatter(truc_mean.index,truc_mean["nb_obs1"],c="lime")
plt.legend(["Hard","Easy"],fontsize=15)
plt.xlabel("User experience",fontsize=15)
plt.ylabel("Average number of observations",fontsize=15)

#%%

plt.scatter(truc_mean.index[0:80],truc_mean["nb_obs0"][0:80],c="red",marker="x")
plt.scatter(truc_mean.index[0:80],truc_mean["nb_obs1"][0:80],c="lime")
plt.legend(["Hard","Easy"])
plt.xlabel("User experience")
plt.ylabel("Average number of observations")


# In[27]:


ordered_bipartite_net = bipartite_net[np.argsort(W1),:][:,np.argsort(W2)]
ordered_W4 = W4[np.argsort(W2)]

ordered_bipartite_net

from matplotlib import colors

cmap = colors.ListedColormap(["white","red"])

color_map = {0: np.array([255, 255, 255]), 
             1: np.array([255, 0 , 0]), # red
             2: np.array([0, 255, 0])} # green

data_3d = np.ndarray(shape=(n01,n02, 3), dtype=int)
for i in range(0, n01):
    for j in range(0, n02):
        data_3d[i][j] = color_map[(ordered_bipartite_net[i,j])*(ordered_W4[j]+1)]
plt.imshow(data_3d)
plt.title("$B'_0$ : True underlying plant-pollinator network",fontsize=15)
plt.xlabel("Insect species",fontsize=13)
plt.ylabel("Plant species",fontsize=13)

#%%
schema = np.array([[0., 0., 1, 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
        0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
        0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0.,
        0., 0.],
       [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0.,
        0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
        0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
        0., 0.],
       [0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 1., 0., 0.,
        0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
        0., 0., 0., 0., 0., 0., 1., 1., 0., 0., 0., 0., 0., 0., 0., 0.,
        0., 0.]])
schema.sum(1)
#plt.imshow(schema,cmap="Greys",interpolation="none")
plt.pcolormesh(schema, edgecolors='k', linewidth=2,cmap="Greys")
plt.tick_params(left = False, right = False , labelleft = False , 
                labelbottom = False, bottom = False)


plt.text(-20,0.5,"$n_{obs_1} = 2$", fontsize = 22)
plt.text(-20,1.5,"$n_{obs_2} = 1$", fontsize = 22)
plt.text(-20,2.5,"$n_{obs_3} = 5$", fontsize = 22)
plt.text(-20,-0.5,"Number of \nobservations", fontsize = 20)
plt.text(3.5,-0.5,"B : Session-pollinator network", fontsize = 20)
#plt.text(-40,-0.5,"Observed\nplant", fontsize = 20)
#plt.text(-40,0.5,"$Flower 4$", fontsize = 20)
#plt.text(-40,1.5,"$Flower 7$", fontsize = 20)
#plt.text(-40,2.5,"$Flower 3$", fontsize = 20)
#plt.text(-35,3.5,"$...$", fontsize = 22,rotation=90)
plt.text(25,3.5,"$...$", fontsize = 22,rotation=90)
plt.text(-15,3.5,"$...$", fontsize = 22,rotation=90)
#plt.text(55,1.5,"$...$", fontsize = 22)



plt.gca().invert_yaxis()

# %%

#plt.imshow(1*(bipartite_obs==0)[np.argsort(W1),:][:,np.argsort(W2)],interpolation="none",cmap="gray")
plt.imshow(1*(bipartite_obs==0)[np.argsort(W1),:][:,np.argsort(W2)],interpolation="none",cmap="gray")
plt.title("$B'$ : observed plant-pollinator network",fontsize=15)
plt.xlabel("Insect species",fontsize=13)
plt.ylabel("Plant species",fontsize=13)
# %%
figure, axis = plt.subplots(2, 2) 


axis[0, 0].scatter(truc_mean.index,truc_mean["nb_obs0"],c="red",marker="x")
axis[0, 0].scatter(truc_mean.index,truc_mean["nb_obs1"],c="lime")
axis[0, 0].legend(["Hard","Easy"])
axis[0, 0].set_xlabel("User experience")
axis[0, 0].set_ylabel("Average number of observations")

axis[1, 0].imshow(data_3d)
axis[1, 0].set_title("$B'_0$ : True underlying plant-insect network")

axis[0, 1].pcolormesh(schema, edgecolors='k', linewidth=2,cmap="Greys")
axis[0, 1].tick_params(left = False, right = False , labelleft = False , 
                labelbottom = False, bottom = False)


axis[0, 1].text(-20,0.5,"$n_{obs_1} = 2$", fontsize = 22)
axis[0, 1].text(-20,1.5,"$n_{obs_2} = 1$", fontsize = 22)
axis[0, 1].text(-20,2.5,"$n_{obs_3} = 5$", fontsize = 22)
axis[0, 1].text(-40,-0.5,"Observed\nplant", fontsize = 20)
axis[0, 1].text(-20,-0.5,"Number of \nobservations", fontsize = 20)
axis[0, 1].text(5,-0.5,"B : Observation-pollinator network", fontsize = 20)
axis[0, 1].text(-40,0.5,"$Flower 4$", fontsize = 20)
axis[0, 1].text(-40,1.5,"$Flower 7$", fontsize = 20)
axis[0, 1].text(-40,2.5,"$Flower 3$", fontsize = 20)
axis[0, 1].text(25,3.5,"$...$", fontsize = 22,rotation=90)
axis[0, 1].text(-15,3.5,"$...$", fontsize = 22,rotation=90)
axis[0, 1].text(-35,3.5,"$...$", fontsize = 22,rotation=90)
axis[0, 1].text(55,1.5,"$...$", fontsize = 22)
axis[0, 1].invert_yaxis()

axis[1, 1].imshow(1*(bipartite_obs==0)[np.argsort(W1),:][:,np.argsort(W2)],interpolation="none",cmap="gray")
axis[1, 1].set_title("$B'$ : observed plant-insect network")
figure.subplots_adjust(left=0.1,
                    bottom=0.1, 
                    right=0.9, 
                    top=0.9, 
                    wspace=0.4, 
                    hspace=0.4)

#%%

plot1 = plt.subplot2grid((3, 3), (0, 0), rowspan=1) 
plot2 = plt.subplot2grid((3, 3), (0, 1), rowspan=2, colspan=2) 
plot3 = plt.subplot2grid((3, 3), (1, 0), rowspan=1) 
plot4 = plt.subplot2grid((3, 3), (1, 1), rowspan=1) 










#%%

args.input_dim1 = n01
args.input_dim2 = n02


# ## Fit
# 

# In[136]:


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




# In[138]:


S0= torch.Tensor(user_exp).reshape(-1,1)
S = S0.clone()
S[:,0] = torch.log10(S0[:,0])
S = (S0-S0.mean(0))/S0.std(0)
#S = S0.clone()
import args


# In[139]:


# init model and optimizer

#torch.manual_seed(2)
#model = VBGAE2(adj_norm,species_index)
model = VBGAE3(adj_norm,species_index,2)

init_parameters(model)

optimizer = Adam(model.parameters(), lr=args.learning_rate)

# train model
roclist = []
loss_list= []


#torch.manual_seed(1)
pbar = tqdm(range(2*int(args.num_epoch)),desc = "Training Epochs")
for epoch in pbar:
    t = time.time()

    A_pred,A_pred2,Z1,Z2,Z3 = model(features1,features2)
    optimizer.zero_grad()
    loss  = 2*norm2*F.binary_cross_entropy(A_pred2.view(-1), torch.Tensor(bipartite).view(-1),weight = weight_tensor2)
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


# In[141]:


A_pred,A_pred2,Z1,Z2,Z3 = model(features1,features2)

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



TEST_EDGES2 = (np.concatenate((test_edges2[0], test_edges_false2[0])),
               np.concatenate((test_edges2[1], test_edges_false2[1])))

proba = (P[W1].T[W2]).T
print(scipy.stats.spearmanr(proba[bipartite_obs==0].reshape(-1,1),
                      A_pred3[bipartite_obs==0].reshape(-1,1)))

print(scipy.stats.spearmanr(proba.reshape(-1,1),
                      A_pred3.reshape(-1,1)))

print(scipy.stats.spearmanr(proba[TEST_EDGES2].reshape(-1,1),
                      A_pred3[TEST_EDGES2].reshape(-1,1)))


# In[ ]:




# In[38]:


plt.imshow((A_pred3)[np.argsort(W1),:][:,np.argsort(W2)],interpolation="none")


# In[39]:




np.round(np.corrcoef(model.mean1.detach().numpy().T,S.T),3)


# In[47]:


stat1 = HSIC_stat(model.mean1,S)
x = np.linspace(0, 0.4, 100)
y = stats.gamma.cdf(x,stat1[3].item(),scale=stat1[4].item())
plt.plot(x,y)
plt.axvline(x = stat1[0].item()*n)
stats.gamma.sf(stat1[0].item()*n, stat1[3].item(), scale=stat1[4].item())


# In[48]:


def plot_latent_space(latent_space1,latent_space2,coloration):
    
    fig,axs = plt.subplots(args.hidden2_dim1-1,args.hidden2_dim1-1,figsize = (15,15))

    for i in range(args.hidden2_dim1,):
        for j in range(i+1,args.hidden2_dim1,):
            A=axs[i,j-1].scatter(latent_space1[:,i],latent_space1[:,j],
                        label="Collection",s=3,
                        c=coloration)
            axs[i,j-1].scatter(latent_space2[:,i],latent_space2[:,j],
                        s=5,label="Observed insects",
                        marker="^",c="r")


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
    


# In[49]:


latent_space1 = model.mean1.detach().numpy()
latent_space2 = model.mean2.detach().numpy()

plot_latent_space(latent_space1,latent_space2,np.log10(S0[:,0].numpy()))
#plot_latent_space(latent_space1,latent_space2,S0[:,0].numpy())


# In[50]:


plot_latent_space(latent_space1,latent_space2,W1[species_index0])


# In[81]:

S0= torch.Tensor(user_exp).reshape(-1,1)
S = S0.clone()
S[:,0] = torch.log10(S0[:,0])
S = (S0-S0.mean(0))/S0.std(0)

#torch.manual_seed(4)
#model2 = VBGAE2(adj_norm,species_index)
model2 = VBGAE3(adj_norm,species_index,2)
init_parameters(model2)

optimizer = Adam(model2.parameters(), lr=args.learning_rate)

# train model2
roclist = []
loss_list= []


#torch.manual_seed(3)
pbar = tqdm(range(2*int(args.num_epoch)),desc = "Training Epochs")
for epoch in pbar:
    t = time.time()

    A_pred,A_pred2,Z1,Z2,Z3 = model2(features1,features2)
    optimizer.zero_grad()
    loss  =2* norm2*F.binary_cross_entropy(A_pred2.view(-1), torch.Tensor(bipartite).view(-1),weight = weight_tensor2)
    loss += norm*F.binary_cross_entropy(A_pred.view(-1), adj_label.to_dense().view(-1), weight = weight_tensor)
    kl_divergence = 0.5/ A_pred.size(0) *( (1 + 2*model2.logstd1 - model2.mean1**2 - torch.exp(model2.logstd1)**2).sum(1).mean()+
                                          (1 + 2*model2.logstd2 - model2.mean2**2 - torch.exp(model2.logstd2)**2).sum(1).mean())
    loss -= kl_divergence
    #independance =torch.log(RFF_HSIC(model2.mean1,S))
    independance = 0.25*n*RFF_HSIC(model2.mean1,S)
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


# In[116]:


A_pred,A_pred2,Z1,Z2,Z3 = model2(features1,features2)

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

TEST_EDGES2 = (np.concatenate((test_edges2[0], test_edges_false2[0])),
               np.concatenate((test_edges2[1], test_edges_false2[1])))

proba = (P[W1].T[W2]).T
print(scipy.stats.spearmanr(proba[bipartite_obs==0].reshape(-1,1),
                      A_pred3[bipartite_obs==0].reshape(-1,1)))

print(scipy.stats.spearmanr(proba.reshape(-1,1),
                      A_pred3.reshape(-1,1)))

print(scipy.stats.spearmanr(proba[TEST_EDGES2].reshape(-1,1),
                      A_pred3[TEST_EDGES2].reshape(-1,1)))

# In[119]:




np.round(np.corrcoef(model2.mean1.detach().numpy().T,S.T),3)


# In[95]:


stat1 = HSIC_stat(model2.mean1,S)
x = np.linspace(0, 0.4, 100)
y = stats.gamma.cdf(x,stat1[3].item(),scale=stat1[4].item())
plt.plot(x,y)
plt.axvline(x = stat1[0].item()*n)
stats.gamma.sf(stat1[0].item()*n, stat1[3].item(), scale=stat1[4].item())


# In[96]:


latent_space1 = model2.mean1.detach().numpy()
latent_space2 = model2.mean2.detach().numpy()
plot_latent_space(latent_space1,latent_space2,np.log10(S0[:,0].numpy()))


# In[97]:


plot_latent_space(latent_space1,latent_space2,W1[species_index0])


# In[ ]:






