#%%
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

#%%
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
#%%
# init model and optimizer

torch.manual_seed(1)
model = VBGAE3(adj_norm,species_index,2)
adv = Adversary(1,16)
init_parameters(model)
init_parameters(adv)


optimizer = Adam(model.parameters(), lr=args.learning_rate)
adv_optimizer = Adam(adv.parameters(),lr = 0.01)

roclist = []
loss_list= []

#%%
torch.manual_seed(1)
pbar = tqdm(range(100),desc = "Training GVAE")
for epoch in pbar:

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

pbar = tqdm(range(1000),desc = "Training adversary")
for epoch in pbar:
    adv_optimizer.zero_grad()
    #A_pred,A_pred2,Z1,Z2,Z3 = model(features1,features2)
    s_hat = adv(model.mean1.detach())
    adv_loss = -torch.abs(torch.corrcoef(torch.cat([S,s_hat],axis=1).T)[0,1])
    adv_loss.backward()
    adv_optimizer.step()
    
    pbar.set_postfix({"adv_loss=": "{:.5f}".format(adv_loss.item())})
        

pbar = tqdm(range(900),desc = "Adversarial training")
for epoch in pbar:
    #Train adversary
    adv_optimizer.zero_grad()
    s_hat = adv(model.mean1.detach())
    adv_loss = -torch.abs(torch.corrcoef(torch.cat([S,s_hat],axis=1).T)[0,1])
    adv_loss.backward()
    adv_optimizer.step()
    
    #Train GVAE
    A_pred,A_pred2,Z1,Z2,Z3 = model(features1,features2)
    optimizer.zero_grad()
    loss  = norm2*F.binary_cross_entropy(A_pred2.view(-1), torch.Tensor(bipartite).view(-1),weight = weight_tensor2)
    loss += norm*F.binary_cross_entropy(A_pred.view(-1), adj_label.to_dense().view(-1), weight = weight_tensor)
    kl_divergence = 0.5/ A_pred.size(0) *( (1 + 2*model.logstd1 - model.mean1**2 - torch.exp(model.logstd1)**2).sum(1).mean()+
                                          (1 + 2*model.logstd2 - model.mean2**2 - torch.exp(model.logstd2)**2).sum(1).mean())
    loss -= kl_divergence
    s_hat = adv(model.mean1)
    adv_loss = -torch.abs(torch.corrcoef(torch.cat([S,s_hat],axis=1).T)[0,1])
    loss -= adv_loss*100
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
#%%
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

cor2 = torch.linalg.norm(torch.corrcoef(torch.cat([model.mean1,S],axis=1).T)[-1,:-1]).item()

print("cor2 : ", cor2)

stat1 = HSIC_stat(model.mean1,S)

x = np.linspace(0, 0.4, 100)
y = stats.gamma.cdf(x,stat1[3].item(),scale=stat1[4].item())
plt.plot(x,y)
plt.axvline(x = stat1[0].item()*n)
stats.gamma.sf(stat1[0].item()*n, stat1[3].item(), scale=stat1[4].item())
# %%

adv_result = pandas.DataFrame(columns = ["AUC","AP",
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
    
    
    list_model =  [VBGAE3(adj_norm,species_index,2) for k in range(1)]
    list_adv =  [Adversary(1) for k in range(1)]

    list_val_roc = []
    for model,adv in zip(list_model,list_adv):
    
        init_parameters(model)
        init_parameters(adv)
        optimizer = Adam(model.parameters(), lr=args.learning_rate)
        adv_optimizer = Adam(adv.parameters(),lr = 0.01)

        # train model
      
        pbar = tqdm(range(100),desc = "Training GVAE")
        for epoch in pbar:

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

        pbar = tqdm(range(1000),desc = "Training adversary")
        for epoch in pbar:
            adv_optimizer.zero_grad()
            s_hat = adv(model.mean1.detach())
            adv_loss = -torch.abs(torch.corrcoef(torch.cat([S,s_hat],axis=1).T)[0,1])
            adv_loss.backward()
            adv_optimizer.step()
            
            pbar.set_postfix({"adv_loss=": "{:.5f}".format(adv_loss.item())})
                

        pbar = tqdm(range(900),desc = "Adversarial training")
        for epoch in pbar:
            #Train adversary
            adv_optimizer.zero_grad()
            s_hat = adv(model.mean1.detach())
            adv_loss = -torch.abs(torch.corrcoef(torch.cat([S,s_hat],axis=1).T)[0,1])
            adv_loss.backward()
            adv_optimizer.step()
            
            #Train GVAE
            A_pred,A_pred2,Z1,Z2,Z3 = model(features1,features2)
            optimizer.zero_grad()
            loss  = norm2*F.binary_cross_entropy(A_pred2.view(-1), torch.Tensor(bipartite).view(-1),weight = weight_tensor2)
            loss += norm*F.binary_cross_entropy(A_pred.view(-1), adj_label.to_dense().view(-1), weight = weight_tensor)
            kl_divergence = 0.5/ A_pred.size(0) *( (1 + 2*model.logstd1 - model.mean1**2 - torch.exp(model.logstd1)**2).sum(1).mean()+
                                                (1 + 2*model.logstd2 - model.mean2**2 - torch.exp(model.logstd2)**2).sum(1).mean())
            loss -= kl_divergence
            s_hat = adv(model.mean1)
            adv_loss = -torch.abs(torch.corrcoef(torch.cat([S,s_hat],axis=1).T)[0,1])
            loss -= adv_loss*100
            loss.backward()
            optimizer.step()
            
            val_roc, val_ap = get_scores(val_edges, val_edges_false, A_pred)
            val_roc2, val_ap2 = get_scores(val_edges2, val_edges_false2, A_pred2)
            
            roclist.append(val_roc2)
            loss_list.append(loss.item())

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
    
    adv_result.iloc[k] = [test_roc,test_ap,
                      test_roc2, test_ap2,
                      test_roc3, test_ap3,
                      stat1[0].detach().numpy() ,p005,p005<0.05,cor2]
    print(adv_result.iloc[k])
    adv_result.to_csv("spipoll_results/spipoll_adv_result.csv")
    
# %%
