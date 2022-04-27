#%%
import pandas as pd
import os

import matplotlib.pyplot as plt
import seaborn as sns

#%%

df = pd.read_csv('data/data_533.csv')


##

df_melt = pd.melt(df.reset_index(), value_vars=['A', 'B', 'C', 'D'])
#df_melt.columns = ['index', 'treatments', 'value']

ax = sns.boxplot(x='variable', y='value', data=df_melt, color='#99c2a2')
ax = sns.swarmplot(x="variable", y="value", data=df_melt, color='#7d0013')
plt.show()


#%%
f = plt.figure(figsize=(19, 15))
plt.matshow(df.corr(), fignum=f.number,cmap ='viridis')
plt.xticks(range(len(df.columns)),df.columns, rotation=45) 
plt.yticks(range(len(df.columns)),df.columns) 
cb = plt.colorbar()
cb.ax.tick_params(labelsize=14)
plt.title('Correlation Matrix', fontsize=16);

# %%
import seaborn as sns
sns.pairplot(df.iloc[:,1:],palette='tab10')
# %%
