### CONFIGS ###
import pandas 

S1 = []
S2 = []

#adj0=pandas.read_csv("data/net.csv",header=0,sep="\t").to_numpy(dtype=float)
#features01 = pandas.read_csv("data/features.csv",header=0,sep="\t")
#species01 = pandas.read_csv("data/species.csv",header=0,sep="\t")

#input_dim1 = species01.shape[1] + features01.shape[1] 
#input_dim2 = adj0.shape[1]
input_dim1 = 83
input_dim2 = 306
hidden1_dim1 = 48
hidden1_dim2 = 32


#hidden1_2dim1 = 8

hidden2_dim1 = 4
hidden2_dim2 = hidden2_dim1

use_feature = True

num_epoch = 1000
learning_rate = 0.005
