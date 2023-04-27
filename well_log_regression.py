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

# def remove_outlier_lof(data, n_neighbors = None, contamination = None, index = None):
#     from sklearn.neighbors import LocalOutlierFactor
    
#     if n_neighbors is None:
#         n_neighbors = 20

#     if contamination is None:
#         contamination = 0.1        
    
#     lof = LocalOutlierFactor(n_neighbors=n_neighbors,contamination=contamination)
    
#     if index is not None:
#         data = np.column_stack([index, data])
#         lof_detect = lof.fit_predict(np.array(data[:,1:]))
#     else:
#         lof_detect = lof.fit_predict(data[:,1:])
        
        
    
#     n_columns_data = data.shape[1]
#     data_no_outlier = np.empty(shape=(0,n_columns_data))
    
    
#     for i in range(len(lof_detect)):
#         if lof_detect[i] == 1:
#             data_no_outlier = np.append(data_no_outlier, np.array(data)[i,:].reshape(1,-1),axis=0)
            
#     return data_no_outlier


#%% Loading DataSet

#Alaska dataSet
alaska2 = lasio.read('well_logs/dados_alaska/dado2.las').df().loc[:,['RHOB','NPHI', 'DT', 'GR', 'ILD']].dropna(subset=['RHOB','NPHI', 'DT', 'GR', 'ILD'])
alaska3 = lasio.read('well_logs/dados_alaska/dado3.las').df().loc[:,['RHOB','NPHI', 'DT', 'GR', 'ILD']].dropna(subset=['RHOB','NPHI', 'DT', 'GR', 'ILD'])
alaska4 = lasio.read('well_logs/dados_alaska/dado4.las').df().loc[:,['RHOB','NPHI', 'DT', 'GR', 'ILD']].dropna(subset=['RHOB','NPHI', 'DT', 'GR', 'ILD'])
alaska5 = lasio.read('well_logs/dados_alaska/dado5.las').df().loc[:,['RHOB','NPHI', 'DT', 'GR', 'ILD']].dropna(subset=['RHOB','NPHI', 'DT', 'GR', 'ILD'])

#%% Remove outliers in Alaska 2
seaborn.pairplot(alaska2)
#%%
alaska2['LOF'] = 1
alaska2.loc[(alaska2['RHOB'] < 2.2) | (alaska2['RHOB'] > 2.55), 'LOF'] = -1
alaska2.loc[(alaska2['NPHI'] < 20) | (alaska2['NPHI'] > 55), 'LOF'] = -1
alaska2.loc[(alaska2['DT'] < 75) | (alaska2['DT'] > 150), 'LOF'] = -1
alaska2.loc[(alaska2['GR'] < 50) | (alaska2['GR'] > 120), 'LOF'] = -1
alaska2.loc[(alaska2['ILD'] < 1) | (alaska2['ILD'] > 22), 'LOF'] = -1
# alaska2.loc[alaska2['ILD'] > 25, 'LOF'] = -1
# alaska2.loc[alaska2['DT'] < 75, 'LOF'] = -1
# alaska2.loc[alaska2['DT'] > 150, 'LOF'] = -1
# alaska2.loc[alaska2['NPHI'] > 55, 'LOF'] = -1

seaborn.pairplot(alaska2, hue='LOF', palette='Dark2')

#%%
alaska2 = alaska2.drop(alaska2[alaska2['LOF']==-1].index)
alaska2 = alaska2.drop(['LOF'], axis = 1)
seaborn.pairplot(alaska2, palette='Dark2')
alaska2.corr()

#%%
lof = LocalOutlierFactor(n_neighbors=10)
y_alaska2_lof = lof.fit_predict(alaska2)
alaska2['LOF'] = y_alaska2_lof
plt.figure(dpi=300)
seaborn.pairplot(alaska2,hue='LOF', palette='Dark2', diag_kind= 'hist')



#%% Remove outliers in Alaska 3
seaborn.pairplot(alaska3)
alaska3['LOF'] = 1
# alaska2.loc[alaska2['RHOB'] < 2.2, 'LOF'] = -1
alaska3.loc[alaska3['RHOB'] > 2.7, 'LOF'] = -1
alaska3.loc[alaska3['ILD'] > 17.5, 'LOF'] = -1
alaska3.loc[alaska3['GR'] > 150, 'LOF'] = -1
# alaska2.loc[alaska2['DT'] < 75, 'LOF'] = -1
alaska3.loc[alaska3['DT'] > 200, 'LOF'] = -1
alaska3.loc[(alaska3['GR'] > 100) & (alaska3['ILD'] > 12.5), 'LOF'] = -1
alaska3.loc[(alaska3['GR'] < 85) & (alaska3['RHOB'] < 2.15), 'LOF'] = -1
alaska3.loc[(alaska3['DT'] > 150) & (alaska3['GR'] > 125), 'LOF'] = -1
alaska3.loc[(alaska3['DT'] > 150) & (alaska3['GR'] > 115), 'LOF'] = -1
alaska3.loc[(alaska3['RHOB'] < 2.2) & (alaska3['GR'] > 150), 'LOF'] = -1
alaska3.loc[(alaska3['NPHI'] > 40) & (alaska3['RHOB'] > 2.60), 'LOF'] = -1
alaska3.loc[(alaska3['NPHI'] < 20) & (alaska3['DT'] > 115), 'LOF'] = -1
alaska3.loc[(alaska3['ILD'] > 11) & (alaska3['NPHI'] > 33), 'LOF'] = -1
alaska3.loc[(alaska3['DT'] > 100) & (alaska3['DT'] < 150) & (alaska3['GR'] > 120), 'LOF'] = -1
# alaska2.loc[alaska2['NPHI'] > 55, 'LOF'] = -1

seaborn.pairplot(alaska3, hue='LOF', palette='Dark2')
#
alaska3 = alaska3.drop(alaska3[alaska3['LOF']==-1].index)
alaska3 = alaska3.drop(['LOF'], axis = 1)
seaborn.pairplot(alaska3, palette='Dark2')

#%%
seaborn.pairplot(alaska5)
alaska5['LOF'] = 1
alaska5.loc[(alaska5['NPHI'] > 50) & (alaska5['DT'] < 120), 'LOF'] = -1
alaska5.loc[(alaska5['NPHI'] > 50), 'LOF'] = -1
alaska5.loc[(alaska5['NPHI'] > 35) & (alaska5['ILD'] > 20), 'LOF'] = -1
alaska5.loc[(alaska5['DT'] < 75) & (alaska5['GR'] > 110), 'LOF'] = -1
alaska5.loc[(alaska5['DT'] < 75) & (alaska5['GR'] < 60), 'LOF'] = -1
alaska5.loc[(alaska5['DT'] < 60) & (alaska5['NPHI'] < 30), 'LOF'] = -1
alaska5.loc[(alaska5['ILD'] > 25), 'LOF'] = -1
alaska5.loc[(alaska5['ILD'] < 5), 'LOF'] = -1
alaska5.loc[(alaska5['GR'] > 125), 'LOF'] = -1
alaska5.loc[(alaska5['GR'] < 50), 'LOF'] = -1
alaska5.loc[(alaska5['ILD'] > 15) & (alaska5['DT'] < 60), 'LOF'] = -1
seaborn.pairplot(alaska5, hue='LOF', palette='Dark2')
#%%
alaska5 = alaska5.drop(alaska5[alaska5['LOF']==-1].index)
alaska5 = alaska5.drop(['LOF'], axis = 1)
seaborn.pairplot(alaska5, palette='Dark2')

#%%
alaska2['LOG'] = 2
alaska3['LOG'] = 3
# alaska4['LOG'] = 4
alaska5['LOG'] = 5

alaska_concat = pd.concat([alaska2,alaska3,alaska5])
plt.figure(dpi=300)
seaborn.pairplot(alaska_concat,hue='LOG',palette='Dark2')

#%%
# alaska2 = alaska2.drop(['LOG'], axis=1)

# X_alaska2 = alaska2.loc[:,['NPHI', 'DT', 'GR', 'ILD']]
# lof = LocalOutlierFactor(n_neighbors=60)
# y_alaska2_lof = lof.fit_predict(alaska2)
# alaska2['LOF'] = y_alaska2_lof
# plt.figure(dpi=300)
# seaborn.pairplot(alaska2,hue='LOF', palette='Dark2', diag_kind= 'hist')

#%%
alaska_ref = pd.concat([alaska2, alaska3])
X_alaska_ref = np.array(alaska_ref.iloc[:,1:5])
y_alaska_ref = np.array(alaska_ref.iloc[:,0]).reshape(-1,1)

#%% Separação dos conjuntos de dados em treino e teste
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X_alaska_ref, y_alaska_ref, test_size=0.33, random_state=7)


#%% Aplicação do power transform

from sklearn.preprocessing import PowerTransformer

pt = PowerTransformer()

X_train_sclnd = pt.fit_transform(X_train)
X_test_sclnd = pt.fit_transform(X_test)

#%% Treinar modelo de regressão linear

from sklearn.linear_model import LinearRegression

lr_model = LinearRegression().fit(X_train_sclnd, y_train)

y_train_pred = lr_model.predict(X_train_sclnd)
y_test_pred = lr_model.predict(X_test_sclnd)




def r_score(pred, real):
    
    return pd.DataFrame(np.concatenate((pred, real),axis=1)).corr().iloc[0,1]


r_score(y_train_pred,y_train)
r_score(y_test_pred,y_test)
#%%
X_alaska5 = np.array(alaska5.iloc[:,1:5])
y_alaska5 = np.array(alaska5.iloc[:,0]).reshape(-1,1)
X_alaska5_sclnd = pt.transform(X_alaska5)

y_alaska5_pred = lr_model.predict(X_alaska5_sclnd)

r_score(y_alaska5_pred,y_alaska5)

#%% Treinar modelo de SVR

from sklearn.svm import SVR

svr_model = SVR().fit(X_train_sclnd, y_train.ravel())

y_train_svr_pred = svr_model.predict(X_train_sclnd).reshape(-1,1)
y_test_svr_pred = svr_model.predict(X_test_sclnd).reshape(-1,1)




r_score(y_train_svr_pred,y_train)
r_score(y_test_svr_pred,y_test)

#%%

y_alaska5_svr_pred = svr_model.predict(X_alaska5_sclnd).reshape(-1,1)

r_score(y_alaska5_pred,y_alaska5)

#%%

plt.figure(dpi=300,figsize=[10,20])
plt.plot(y_alaska5, alaska5.index)
plt.plot(y_alaska5_pred, alaska5.index)
plt.plot(y_alaska5_svr_pred, alaska5.index)