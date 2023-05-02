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

#%% Loading DataSet

#Alaska dataSet
alaska2 = lasio.read('well_logs/dados_alaska/dado2.las').df().loc[:,['RHOB','NPHI', 'DT', 'GR', 'ILD']].dropna(subset=['RHOB','NPHI', 'DT', 'GR', 'ILD'])
alaska2.corr()
alaska3 = lasio.read('well_logs/dados_alaska/dado3.las').df().loc[:,['RHOB','NPHI', 'DT', 'GR', 'ILD']].dropna(subset=['RHOB','NPHI', 'DT', 'GR', 'ILD'])
alaska4 = lasio.read('well_logs/dados_alaska/dado4.las').df().loc[:,['RHOB','NPHI', 'DT', 'GR', 'ILD']].dropna(subset=['RHOB','NPHI', 'DT', 'GR', 'ILD'])
alaska4.corr()
alaska5 = lasio.read('well_logs/dados_alaska/dado5.las').df().loc[:,['RHOB','NPHI', 'DT', 'GR', 'ILD']].dropna(subset=['RHOB','NPHI', 'DT', 'GR', 'ILD'])
alaska5.corr()

#%% Correlação dos dados sem pre-processamento
#### alaska2
alaska2_corr = alaska2.corr()
fig, ax = plt.subplots()
table_data = [alaska2_corr.columns.to_list()] + alaska2_corr.values.tolist()

ax.axis('off')
ax.axis('tight')
ax.set_title('Matriz de Correlação sem pre-processamento - Alaska2', fontsize=14) # Adicionando um título à imagem
fig.set_dpi(300)
ax.table(cellText=table_data, colLabels=None, cellLoc='center', loc='center')
#### alaska3
alaska3_corr = alaska3.corr()
fig, ax = plt.subplots()
table_data = [alaska3_corr.columns.to_list()] + alaska3_corr.values.tolist()

ax.axis('off')
ax.axis('tight')
ax.set_title('Matriz de Correlação sem pre-processamento - Alaska3', fontsize=14) # Adicionando um título à imagem
fig.set_dpi(300)
ax.table(cellText=table_data, colLabels=None, cellLoc='center', loc='center')

#### alaska5
alaska5_corr = alaska5.corr()
fig, ax = plt.subplots()
table_data = [alaska5_corr.columns.to_list()] + alaska5_corr.values.tolist()

ax.axis('off')
ax.axis('tight')
ax.set_title('Matriz de Correlação sem pre-processamento - Alaska5', fontsize=14) # Adicionando um título à imagem
fig.set_dpi(300)
ax.table(cellText=table_data, colLabels=None, cellLoc='center', loc='center')
#%% Aplicar o covariance

from sklearn.covariance import EllipticEnvelope

cov = EllipticEnvelope()

#### Alaska2
alaska2_cov = cov.fit_predict(alaska2.iloc[:,:3])
alaska2['LOF'] = alaska2_cov
seaborn.pairplot(alaska2, hue='LOF', palette='Dark2').fig.suptitle("Alaska 2 - Covariance", y=1.05, fontsize=20)

plt.figure(dpi=300, figsize=[20,20])
plt.subplot(151)
plt.scatter(alaska2['RHOB'], alaska2.index, c=alaska2_cov)
plt.subplot(152)
plt.scatter(alaska2['NPHI'], alaska2.index, c=alaska2_cov)
plt.subplot(153)
plt.scatter(alaska2['DT'], alaska2.index, c=alaska2_cov)
plt.subplot(154)
plt.scatter(alaska2['GR'], alaska2.index, c=alaska2_cov)
plt.subplot(155)
plt.scatter(alaska2['ILD'], alaska2.index, c=alaska2_cov)
plt.xlim(0,40)

alaska2 = alaska2.drop(alaska2[alaska2['LOF']==-1].index)
alaska2 = alaska2.drop(['LOF'], axis = 1)
alaska2.corr()
seaborn.pairplot(alaska2, palette='Dark2')

#### Alaska3
alaska3_cov = cov.fit_predict(alaska3)
alaska3['LOF'] = alaska3_cov
seaborn.pairplot(alaska3, hue='LOF', palette='Dark2').fig.suptitle("Alaska 2 - Covariance", y=1.05, fontsize=20)

plt.figure(dpi=300, figsize=[20,20])
plt.subplot(151)
plt.scatter(alaska3['RHOB'], alaska3.index, c=alaska3_cov)
plt.subplot(152)
plt.scatter(alaska3['NPHI'], alaska3.index, c=alaska3_cov)
plt.subplot(153)
plt.scatter(alaska3['DT'], alaska3.index, c=alaska3_cov)
plt.subplot(154)
plt.scatter(alaska3['GR'], alaska3.index, c=alaska3_cov)
plt.subplot(155)
plt.scatter(alaska3['ILD'], alaska3.index, c=alaska3_cov)
plt.xlim(0,40)

alaska3 = alaska3.drop(alaska3[alaska3['LOF']==-1].index)
alaska3 = alaska3.drop(['LOF'], axis = 1)
alaska3.corr()
seaborn.pairplot(alaska3, palette='Dark2')

#### Alaska5
alaska5_cov = cov.fit_predict(alaska5)
alaska5['LOF'] = alaska5_cov
seaborn.pairplot(alaska5, hue='LOF', palette='Dark2').fig.suptitle("Alaska 2 - Covariance", y=1.05, fontsize=20)

plt.figure(dpi=300, figsize=[20,20])
plt.subplot(151)
plt.scatter(alaska5['RHOB'], alaska5.index, c=alaska5_cov)
plt.subplot(152)
plt.scatter(alaska5['NPHI'], alaska5.index, c=alaska5_cov)
plt.subplot(153)
plt.scatter(alaska5['DT'], alaska5.index, c=alaska5_cov)
plt.subplot(154)
plt.scatter(alaska5['GR'], alaska5.index, c=alaska5_cov)
plt.subplot(155)
plt.scatter(alaska5['ILD'], alaska5.index, c=alaska5_cov)
plt.xlim(0,40)

alaska5 = alaska5.drop(alaska5[alaska5['LOF']==-1].index)
alaska5 = alaska5.drop(['LOF'], axis = 1)
alaska5.corr()
seaborn.pairplot(alaska5, palette='Dark2')

#%% Remove outliers in Alaska 2
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

y_alaska5_pred = svr_model.predict(X_alaska5_sclnd).reshape(-1,1)

r_score(y_alaska5_pred,y_alaska5)

