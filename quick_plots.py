#%%
import numpy as np
import matplotlib.pyplot as plt

#%%
a = np.random.rand(200)*10
b = np.sin(a)
c = np.random.normal(loc = 0, scale=a)
d = a**4
e = 1/a


axs, fig = plt.subplots(2,2)
y_axs = [b,c,d,e]
for i,ax in enumerate(axs):
    ax

plt.scatter(a,f)

