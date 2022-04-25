#%%
import numpy as np
import matplotlib.pyplot as plt
import csv
import pandas as pd

#%%

data = pd.read_csv("data/actual/100.csv")
data = data.iloc[:,1:]
vars = list(data.columns)
n_vars = len(vars)

#%%
# Pairwise plots
fig, axs = plt.subplots(n_vars,n_vars,figsize=(20,20))

for i,axs_i in enumerate(axs):
    for j,ax in enumerate(axs_i):
        if i != j:
            ax.scatter(data[vars[i]],data[vars[j]],linewidths =0.1)
            ax.set_title(f'{vars[i]} vs {vars[j]}')

#%%
# Normal hists
fig, axs = plt.subplots(n_vars,figsize=(15,15))
print(type(axs))
for i,ax in enumerate(axs):
    ax.hist(data[vars[i]])
    ax.set_title(vars[i], fontsize = 20)


#%%
# Linear Correlation
corre_matrix = np.zeros((n_vars,n_vars))
for i in range(n_vars):
    for j in range(n_vars):
        corre_matrix[i,j] = np.round(np.corrcoef(data[vars[i]],data[vars[j]])[0,1],3)
    

corre_matrix


# %%



def MI(x,y,Nbins=21, names=None):
    bins = np.linspace(np.min(x),np.max(x),Nbins)
    eps=np.spacing(1)
    x_marginal = np.histogram(x,bins=bins)[0]
    x_marginal = x_marginal/x_marginal.sum()
    y_marginal = np.array(np.histogram(y,bins=bins)[0])
    y_marginal = y_marginal/y_marginal.sum()
    xy_joint = np.array(np.histogram2d(x,y,bins=(bins,bins))[0])
    xy_joint = xy_joint/xy_joint.sum()
    plt.figure()
    plt.subplot(1,2,1)
    plt.imshow(xy_joint.T,origin='lower')
    plt.title('joint')
    plt.subplot(1,2,2)
    plt.imshow((x_marginal[:,None]*y_marginal[None,:]).T,origin='lower')
    plt.title('product of marginals')
    mi=np.sum(xy_joint*np.log(xy_joint/(x_marginal[:,None]*y_marginal[None,:]+eps)+eps))
    if names is None:
        plt.suptitle(f'Mutual information: {mi}')
    else:
        plt.suptitle(f'Mutual information between {names[0]}, {names[1]}: {mi:.3f}')
    return(mi)
MI(data['A'],data['F'], names=['A','F'])
for i in vars:
    for j in vars:
        MI(data[i], data[j], names=[i,j])



# %%
