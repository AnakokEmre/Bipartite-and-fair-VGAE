# Bipartite-and-fair-VGAE
Bipartite and fair adaptation of Kipf's VGAE in python. 

Code for the simulated part is in the simulation folder.
Code for replication of the Spipoll sampling process is in the spipoll_simulation folder.
Application on the Spipoll dataset, from 2016 to 2020, is available in the spipoll folder.

# A. The Linear Embedding Case

## A.1 Overview
Let $X$ a $n\times d$ matrix and let $S$ be $n \times d_s$ matrix. Without loss of generality, we assume that each column of $X$ has been centered. We wish to perform a one dimensional principal component analysis on $X$ that would yield us a vector $v$ and a lower dimensional embedding of $X$ given by $Xv$ that maximizes the variance. However, we wish to have a latent representation $Xv$ independent of the protected variable $S$.  

If we were in the context of probabilistic PCA (Tipping and Bishop, 1999)  where $X$ and $S$ would have been multivariate Gaussian, 
projecting $X$ onto the space orthogonal to $S$ (written as $S^\perp$) beforehand would have been enough to guarantee the independence between $S$ and the latent representation $Xv$, this can be solved using PPCA with covariates (Kalaitzis and Lawrence, 2012). 

We show that this approach is equivalent to find the optimal projection with respect to an independence constraint.


We note $P_S = S(S^\top S)^{-1}S^\top$ the orthogonal projection on the span of $S$ and  $P_{S^\bot}=I_d - P_SX$ the orthogonal projection on the space orthogonal to the span of $S$. 

