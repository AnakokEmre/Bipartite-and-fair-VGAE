import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.optim import Adam
from tqdm import tqdm
import matplotlib.pyplot as plt
from HSIC import *
from scipy.stats import gamma

import pandas

    
def init_parameters(net):
        for m in net.modules():
            if isinstance(m, nn.Linear):
                # nn.init.xavier_uniform(m.weight.data)
                torch.nn.init.normal_(m.weight.data)
                #nn.init.constant_(m.bias.data, 0)
                
                

class ACP_model0(nn.Module):
    def __init__(self,input_dim,output_dim):
        super(ACP_model0,self).__init__()
        self.dense1_mean = nn.Linear(input_dim,output_dim,bias=False)

        self.dense2 = nn.Linear(output_dim,input_dim,bias=False)
        
    def encode(self, X):
        return  self.dense1_mean(X)

    def forward(self,x):
        y    = self.encode(x)
        xhat = self.dense2(y)
        return y,xhat
 
##################################################
np.random.seed(1)
n=1000

t = np.random.normal(size=(n,1))
s = np.random.normal(size=(n,1))
#s = np.random.randint(0,2,size=n).reshape(-1,1)

K = np.random.normal(size=(2,5))*3

st = np.concatenate([s,t],axis=1)

ST = torch.Tensor(st)
S= torch.Tensor(s)

x=np.dot(st,K)
X = torch.Tensor(x)


#################################################
#PCA

torch.manual_seed(1)
model1 = ACP_model0(5,2)
#init_parameters(model1)

learning_rate = 0.01
epochs=2000
criterion = torch.nn.MSELoss()
optimizer = Adam(model1.parameters(), lr=learning_rate)
torch.manual_seed(1)

pbar = tqdm(range(int(200)),desc = "training Epochs")
for epoch in pbar:
    optimizer.zero_grad()
    latent_space1,reconstruction1=  model1(X)
    loss = criterion(reconstruction1,X)
    loss.backward()
    optimizer.step()
    pbar.set_postfix({"train_loss=": "{:.5f}".format(loss.item())})

    
print(loss)
print(criterion(reconstruction1,X).item())




plt.scatter((latent_space1/latent_space1.std(0)).detach().numpy()[:,0],
            (latent_space1/latent_space1.std(0)).detach().numpy()[:,1],c=s)
cb=plt.colorbar(label="S")
cb.set_label(label="S",size=30,rotation = 0,loc = "center")

plt.scatter(latent_space1.detach().numpy()[:,0],
            latent_space1.detach().numpy()[:,1],c=t)
cb=plt.colorbar(label="T")
cb.set_label(label="T",size=30,rotation = 0,loc = "center")

er1 = '{:0.3e}'.format(criterion(reconstruction1,X).item())
stat1 = HSIC_stat(S,latent_space1)
p1 = gamma.sf(stat1[0].item()*n, stat1[3].item(), scale=stat1[4].item())
print(torch.corrcoef(torch.cat([latent_space1.T,ST.T])))

######PROJ


torch.manual_seed(0)
model2 = ACP_model0(5,2)

learning_rate = 0.01
epochs=2000
criterion = torch.nn.MSELoss()
optimizer = Adam(model2.parameters(), lr=learning_rate)
torch.manual_seed(1)

def projection_on_orthogonal(S,T):
    n= S.shape[0]
    inv = torch.inverse((S.T).matmul(S))
    proj = (torch.eye(n) - S.matmul(inv).matmul(S.T)).matmul(T)
    return proj

Xproj = projection_on_orthogonal(S, X)

pbar = tqdm(range(int(200)),desc = "training Epochs")
for epoch in pbar:
    optimizer.zero_grad()
    latent_space2,reconstruction2=  model2(Xproj)
    loss = criterion(reconstruction2,Xproj) #- 0.5*(1 + 2*model1.logstd - model1.mean**2 - torch.exp(model1.logstd)**2).sum(1).mean()
    loss.backward()
    optimizer.step()
    pbar.set_postfix({"train_loss=": "{:.5f}".format(loss.item())})




plt.scatter(latent_space2.detach().numpy()[:,0],
            latent_space2.detach().numpy()[:,1],c=s)
cb=plt.colorbar(label="S")
cb.set_label(label="S",size=30,rotation = 0,loc = "center")

plt.scatter(latent_space2.detach().numpy()[:,0],
            latent_space2.detach().numpy()[:,1],c=t)
cb=plt.colorbar(label="T")
cb.set_label(label="T",size=30,rotation = 0,loc = "center")

print(criterion(reconstruction2,X).item())


er2 = '{:0.3e}'.format(criterion(reconstruction2,X).item())
stat2 = HSIC_stat(S,latent_space2)
p2 = gamma.sf(stat2[0].item()*n, stat2[3].item(), scale=stat2[4].item())
print(torch.corrcoef(torch.cat([latent_space2.T,ST.T])))





###########HSIC 
torch.manual_seed(1)
model3 = ACP_model0(5,2)
learning_rate = 0.01
criterion = torch.nn.MSELoss()
optimizer = Adam(model3.parameters(), lr=learning_rate)
torch.manual_seed(1)


pbar = tqdm(range(int(6000)),desc = "training Epochs")
for epoch in pbar:
    optimizer.zero_grad()
    latent_space3,reconstruction3=  model3(X)
    loss = criterion(reconstruction3,X) #- 0.5*(1 + 2*model1.logstd - model1.mean**2 - torch.exp(model1.logstd)**2).sum(1).mean()
    independance = RFF_HSIC(latent_space3,S)
    loss += epoch**2* independance
    loss.backward()
    optimizer.step()
    pbar.set_postfix({"train_loss=": "{:.5f}".format(loss.item())})
    


plt.scatter(latent_space3.detach().numpy()[:,0],
            latent_space3.detach().numpy()[:,1],c=s)
plt.colorbar(label="S")

plt.scatter(latent_space3.detach().numpy()[:,0],
            latent_space3.detach().numpy()[:,1],c=t)
plt.colorbar(label="T")
print(criterion(reconstruction3,X).item())

er3 = '{:0.3e}'.format(criterion(reconstruction3,X).item())
stat3 = HSIC_stat(S,latent_space3)
p3 = gamma.sf(stat3[0].item()*n, stat3[3].item(), scale=stat3[4].item())
torch.corrcoef(torch.cat([latent_space3.T,ST.T]))




result = pandas.DataFrame(columns = ["er1","cor1","p1","HSIC1","er2","cor2","p2","HSIC2","er3","cor3","p3","HSIC3"],
                          index = range(100))

test_size = 200
for k in range(100):
    print(k)
    t = np.random.normal(size=(n,1))
    s = np.random.normal(size=(n,1))
    K = np.random.normal(size=(2,5))*3

    st = np.concatenate([s,t],axis=1)

    ST = torch.Tensor(st)
    S= torch.Tensor(s)

    x=np.dot(st,K)
    X = torch.Tensor(x)
    X_train,X_test = X[:-test_size,],X[-test_size:,]
    
    model1 = ACP_model0(5,2)
    learning_rate = 0.01
    criterion = torch.nn.MSELoss()
    optimizer = Adam(model1.parameters(), lr=learning_rate)
    torch.manual_seed(1)

    pbar = tqdm(range(int(200)),desc = "training Epochs")
    for epoch in pbar:
        optimizer.zero_grad()
        latent_space1,reconstruction1=  model1(X_train)
        loss = criterion(reconstruction1,X_train)
        loss.backward()
        optimizer.step()
        pbar.set_postfix({"train_loss=": "{:.5f}".format(loss.item())})
        
    latent_space1,reconstruction1=  model1(X_test)
    er1 =(criterion(reconstruction1,X_test).item())
    stat1 = HSIC_stat(S[-test_size:,],latent_space1)
    p1 = gamma.sf(stat1[0].item()*test_size, stat1[3].item(), scale=stat1[4].item())
    cor1=torch.linalg.norm(torch.corrcoef(torch.cat([latent_space1.T,ST[-test_size:,].T]))[2,:2])
    
    
    model2 = ACP_model0(5,2)
    optimizer = Adam(model2.parameters(), lr=learning_rate)
    Xproj = projection_on_orthogonal(S, X)
    Xproj_train = Xproj[:-test_size,]
    Xproj_test = Xproj[-test_size:,]
    
    pbar = tqdm(range(int(200)),desc = "training Epochs")
    for epoch in pbar:
        optimizer.zero_grad()
        latent_space2,reconstruction2=  model2(Xproj_train)
        loss = criterion(reconstruction2,Xproj_train) 
        loss.backward()
        optimizer.step()
        pbar.set_postfix({"train_loss=": "{:.5f}".format(loss.item())})

    
    latent_space2,reconstruction2=  model2(Xproj_test)
    er2 = criterion(reconstruction2,X_test).item()
    stat2 = HSIC_stat(S[-test_size:,],latent_space2)
    p2 = gamma.sf(stat2[0].item()*test_size, stat2[3].item(), scale=stat2[4].item())
    cor2=torch.linalg.norm(torch.corrcoef(torch.cat([latent_space2.T,ST[-test_size:,].T]))[2,:2])
    
    
    
    
    list_model = [ACP_model0(5,2) for j in range(10)]
    list_independence = []
    for model3 in list_model:
        optimizer = Adam(model3.parameters(), lr=learning_rate)    
        pbar = tqdm(range(int(300)),desc = "training Epochs")
        for epoch in pbar:
            optimizer.zero_grad()
            latent_space3,reconstruction3=  model3(X_train)
            loss = criterion(reconstruction3,X_train) #- 0.5*(1 + 2*model1.logstd - model1.mean**2 - torch.exp(model1.logstd)**2).sum(1).mean()
            independance = RFF_HSIC(latent_space3,S[:-test_size,])
            loss += 10000* independance
            loss.backward()
            optimizer.step()
            pbar.set_postfix({"train_loss=": "{:.5f}".format(loss.item())})
        list_independence.append(independance.item())
        best_model = list_model[np.argmin(list_independence)]
        latent_space3,reconstruction3=  best_model(X_test)
        er3 = criterion(reconstruction3,X_test).item()
        stat3 = HSIC_stat(S[-test_size:,],latent_space3)
        p3 = gamma.sf(stat3[0].item()*test_size, stat3[3].item(), scale=stat3[4].item())
        cor3=torch.linalg.norm(torch.corrcoef(torch.cat([latent_space3.T,ST[-test_size:,].T]))[2,:2])
        
    result.iloc[k] =  [er1,cor1.item(),p1,stat1[0].item(),
                       er2,cor2.item(),p2,stat2[0].item(),
                       er3,cor3.item(),p3,stat3[0].item()]
    print(result.iloc[k])




    
    
    
    
    
    
    
    




