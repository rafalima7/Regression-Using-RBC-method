# -*- coding: utf-8 -*-
"""
Created on Tue Apr 25 16:41:09 2023

@author: Rafael Lima
"""

#%% Load Packages

import lasio
import seaborn
import matplotlib.pyplot as plt
from sklearn.neighbors import LocalOutlierFactor
import pandas as pd
import numpy as np
#%% Function to remove outliers detected by LocalOutlierFactor

def remove_outlier_lof(data, n_neighbors = None, contamination = None, index = None):
    from sklearn.neighbors import LocalOutlierFactor
    
    if n_neighbors is None:
        n_neighbors = 20

    if contamination is None:
        contamination = 0.1        
    
    lof = LocalOutlierFactor(n_neighbors=n_neighbors,contamination=contamination)
    
    if index is not None:
        data = np.column_stack([index, data])
        lof_detect = lof.fit_predict(np.array(data[:,1:]))
    else:
        lof_detect = lof.fit_predict(data[:,1:])
        
        
    
    n_columns_data = data.shape[1]
    data_no_outlier = np.empty(shape=(0,n_columns_data))
    
    
    for i in range(len(lof_detect)):
        if lof_detect[i] == 1:
            data_no_outlier = np.append(data_no_outlier, np.array(data)[i,:].reshape(1,-1),axis=0)
            
    return data_no_outlier


#%% Loading DataSet

#Alaska dataSet
alaska2 = lasio.read('well_logs/dados_alaska/dado2.las').df().loc[:,['RHOB','NPHI', 'DT', 'GR', 'ILD']].dropna(subset=['RHOB','NPHI', 'DT', 'GR', 'ILD'])
alaska3 = lasio.read('well_logs/dados_alaska/dado3.las').df().loc[:,['RHOB','NPHI', 'DT', 'GR', 'ILD']].dropna(subset=['RHOB','NPHI', 'DT', 'GR', 'ILD'])
alaska4 = lasio.read('well_logs/dados_alaska/dado4.las').df().loc[:,['RHOB','NPHI', 'DT', 'GR', 'ILD']].dropna(subset=['RHOB','NPHI', 'DT', 'GR', 'ILD'])
alaska5 = lasio.read('well_logs/dados_alaska/dado5.las').df().loc[:,['RHOB','NPHI', 'DT', 'GR', 'ILD']].dropna(subset=['RHOB','NPHI', 'DT', 'GR', 'ILD'])

#%%
seaborn.pairplot(alaska2)
alaska2['LOF'] = 1
alaska2.loc[alaska2['RHOB'] < 2.2, 'LOF'] = -1
alaska2.loc[alaska2['RHOB'] > 2.6, 'LOF'] = -1
alaska2.loc[alaska2['ILD'] > 25, 'LOF'] = -1
alaska2.loc[alaska2['DT'] < 75, 'LOF'] = -1
alaska2.loc[alaska2['DT'] > 150, 'LOF'] = -1
alaska2.loc[alaska2['NPHI'] > 55, 'LOF'] = -1

seaborn.pairplot(alaska2, hue='LOF', palette='Dark2')

alaska2 = alaska2.drop(alaska2[alaska2['LOF']==-1].index)
alaska2 = alaska2.drop(['LOF'], axis = 1)
seaborn.pairplot(alaska2, palette='Dark2')

#%%
alaska2['LOG'] = 2
alaska3['LOG'] = 3
alaska4['LOG'] = 4
alaska5['LOG'] = 5

alaska_concat = pd.concat([alaska2,alaska3,alaska4,alaska5])
plt.figure(dpi=300)
seaborn.pairplot(alaska_concat,hue='LOG',palette='Dark2')

#%%
# alaska2 = alaska2.drop(['LOG'], axis=1)

# X_alaska2 = alaska2.loc[:,['NPHI', 'DT', 'GR', 'ILD']]
lof = LocalOutlierFactor(n_neighbors=60)
y_alaska2_lof = lof.fit_predict(alaska2)
alaska2['LOF'] = y_alaska2_lof
plt.figure(dpi=300)
seaborn.pairplot(alaska2,hue='LOF', palette='Dark2', diag_kind= 'hist')

#%%
alaska2 = alaska2.drop(alaska2[alaska2['LOF']==-1].index)
alaska2 = alaska2.drop(['LOF'], axis = 1)
seaborn.pairplot(alaska2)
#%%
alaska3 = alaska3.drop(['LOG'], axis=1)
X_alaska3 = alaska3.loc[:, ['NPHI', 'DT', 'GR', 'ILD']]
lof = LocalOutlierFactor(n_neighbors=20,contamination=0.1)
y_alaska2_lof = lof.fit_predict(X_alaska2)
alaska2['LOF'] = y_alaska2_lof
seaborn.pairplot(alaska2,hue='LOF', palette='Dark2')
