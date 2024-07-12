#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 19 16:08:12 2024

@author: mmip
"""

from HSIC import *
import time
from tqdm import tqdm
from sys import getsizeof

def quick_HSIC(X,L,sumL,sumL1):
    n = X.shape[0]
    distX = torch.cdist(X,X)**2
    K =torch.exp(-distX/2)

    HSIC =  (K*L).sum() + K.sum()*sumL/(n**2) -2*(K.sum(0)@sumL1)/n

    
    return HSIC/(n**2)



def RFF_HSIC(Z,S,D=100):
    n= Z.shape[0]
    D=100
    omegaZ = torch.normal(mean=0.,std=1.,size=(D,Z.shape[1]))
    bZ = torch.rand(D) * 2*torch.pi
    
    omegaS = torch.normal(mean=0.,std=1.,size=(D,S.shape[1]))
    bS = torch.rand(D) * 2*torch.pi
    
    Zo=torch.cos((Z@omegaZ.T+bZ))*np.sqrt(2/D)
    So=torch.cos((S@omegaS.T+bS))*np.sqrt(2/D)
    
    HSIC = (Zo.T@So -Zo.sum(0).reshape(-1,1) @ So.sum(0).reshape(1,-1)/n).square().sum()/n**2

    
    
    return HSIC

N = [100,1000,2000,5000,8000,10000]
ntimes = 100


RES0 = pandas.DataFrame(columns=["res0","res1",
                                "time_quick","time_RFF",
                                "time_grad_quick",
                                "time_grad_RFF","n"],index=range(len(N)*ntimes))

RES0["n"] = np.repeat(N,ntimes)

j=0
for n in N:
    S = torch.normal(0, 1,(n,4),requires_grad=True)
    distS = torch.cdist(S,S)**2
    sigmaS = 1
    L = torch.exp(-distS/(2*sigmaS))
    sumL  = L.sum()
    sumL1 = L.sum(1)
    D = int(np.ceil(np.sqrt(n)))
    
    pbar = tqdm(range(ntimes))
    for k in pbar:
        X = torch.normal(0, 1,(n,4),requires_grad=True)
        t0_0 = time.time() 
        res0 = quick_HSIC(X, L, sumL, sumL1)
        t0_1 = time.time()
        time_quick = t0_1-t0_0
        
        t1_0 = time.time() 
        res1 = RFF_HSIC(X,S,D)
        t1_1 = time.time()
        time_RFF = t1_1-t1_0
        
        t0_0 = time.time() 
        res0.backward(retain_graph=True)
        t0_1 = time.time()
        time_grad_quick = t0_1-t0_0
        
        t0_0 = time.time() 
        res1.backward(retain_graph=True)
        t0_1 = time.time()
        time_grad_RFF = t0_1-t0_0
        
        RES0.iloc[j,:-1] = [res0.item(),res1.item(),
                                        time_quick,time_RFF,
                                        time_grad_quick,
                                        time_grad_RFF]
        j=j+1
        



RES0.groupby(["n"]).mean()


RES1 = pandas.DataFrame(columns=["res0","res1",
                                "time_quick","time_RFF",
                                "time_grad_quick",
                                "time_grad_RFF","n"],index=range(len(N)*ntimes))

RES1["n"] = np.repeat(N,ntimes)

j=0
for n in N:
    S = torch.normal(0, 1,(n,4),requires_grad=True)
    distS = torch.cdist(S,S)**2
    sigmaS = 1
    L = torch.exp(-distS/(2*sigmaS))
    sumL  = L.sum()
    sumL1 = L.sum(1)
    D = int(np.ceil(np.sqrt(n)))
    
    pbar = tqdm(range(ntimes))
    for k in pbar:
        X = 3*S.clone()
        t0_0 = time.time() 
        res0 = quick_HSIC(X, L, sumL, sumL1)
        t0_1 = time.time()
        time_quick = t0_1-t0_0
        
        t1_0 = time.time() 
        res1 = RFF_HSIC(X,S,D)
        t1_1 = time.time()
        time_RFF = t1_1-t1_0
        
        t0_0 = time.time() 
        res0.backward(retain_graph=True)
        t0_1 = time.time()
        time_grad_quick = t0_1-t0_0
        
        t0_0 = time.time() 
        res1.backward(retain_graph=True)
        t0_1 = time.time()
        time_grad_RFF = t0_1-t0_0
        
        RES1.iloc[j,:-1] = [res0.item(),res1.item(),
                                        time_quick,time_RFF,
                                        time_grad_quick,
                                        time_grad_RFF]
        j=j+1
        
RES1.groupby(["n"]).mean()


RES=pandas.concat([RES1,RES0])
RES["H"] = np.repeat([0,1],ntimes*len(N))
RES.to_csv("simulation_results/HSIC_timer.csv")

