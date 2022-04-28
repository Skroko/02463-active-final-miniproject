#%%
import numpy as np
import matplotlib.pyplot as plt
import csv
import pandas as pd
import sklearn.feature_selection as sf

#%%

# data = pd.read_csv("Data/data_566.csv")
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
        corre_matrix[i,j] = np.round(np.corrcoef(data[vars[i]]+1e-8,data[vars[j]]+1e-8)[0,1],3)
    

corre_matrix

#%%
# sm.mutual_info_score(data[vars[0]],data[vars[1]])

mutal_score_M = np.zeros((N_vars,N_vars))
for i in range(N_vars):
    for j in range(N_vars):
        x = np.array(data[vars[i]]).reshape((-1,1))
        y = np.array(data[vars[j]])
        mutal_score_M[i,j] = np.round(sf.mutual_info_regression(x,y),3)

mutal_score_M



#%%

a = np.random.uniform(low = 1, high = 22,size = 1000)

b = np.random.normal(np.random.random(1000)*10,a)

c = np.random.uniform(low= -10, high = np.exp(b), size = 1000)

d = np.random.normal(c,np.abs(np.cos(np.random.normal(22,4))))

np.round(sf.mutual_info_regression(a.reshape((-1,1)),d),10)  

#%%

a = np.random.normal(np.random.random(1000)*10,x)
b = np.random.uniform(low = 1, high = 22,size = 1000)

#%%
from scipy import stats as st

a = np.random.normal(np.random.random(1000)*10,22)
b = np.random.normal(np.random.random(1000)*10,4)

st.ttest_ind(a,b)