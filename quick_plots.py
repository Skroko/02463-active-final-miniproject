#%%
import numpy as np
import matplotlib.pyplot as plt
import csv
import pandas as pd

#%%

data = pd.read_csv("Data/data_534.csv")
N_vars = 4
vars = ["A","B","C","D"]

#%%
# Pairwise plots
fig, axs = plt.subplots(4,4,figsize=(15,15))

for i,axs_i in enumerate(axs):
    for j,ax in enumerate(axs_i):
        if i != j:
            ax.scatter(data[vars[i]],data[vars[j]],linewidths =0.1)

#%%
# Normal hists
fig, axs = plt.subplots(4,figsize=(15,15))
vars = ["A","B","C","D"]
print(type(axs))
for i,ax in enumerate(axs):
    ax.hist(data[vars[i]])


#%%
# Linear Correlation
corre_matrix = np.zeros((N_vars,N_vars))
for i in range(N_vars):
    for j in range(N_vars):
        corre_matrix[i,j] = np.round(np.corrcoef(data[vars[i]],data[vars[j]])[0,1],3)
    

corre_matrix

