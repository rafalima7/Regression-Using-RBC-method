# -*- coding: utf-8 -*-
"""
Created on Tue Feb 14 10:57:11 2023

@author: Rafael Lima
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, r2_score as r2, mean_squared_error as mse
import matplotlib.pyplot as plt
from sympy import symbols
from sympy.printing import latex

class RBC: 
    def __init__(self, classifier_method, regression_method):
        
        
        self.classifier_method = classifier_method
        self.regression_method = regression_method
        self.default_params_class = classifier_method.get_params()
        self.default_params_reg = regression_method.get_params()
        
    def fit(self, X, y_class, y_reg, params_class = None, params_reg = None):
        
        if params_class is None:
            params_class = self.default_params_class
        if params_reg is None:
            params_reg = {}
            n_class = len(set(y_class))
            for i in range(1, n_class+1):
                params_reg[i] = self.default_params_reg
        
        self.X = X
        self.y_class = y_class
        self.y_reg = y_reg
        self.classifier_method.set_params(**params_class)
        self.classifier_method.fit(X, y_class)
        
        
        n_class = len(set(y_class))
        self.n_class = n_class
        self.n_features = X.shape[1]
        
        regression_method_dic = {}
        
        for t in range(1, n_class + 1):
        
            regression_method_dic[t] = self.regression_method
            regression_method_dic[t].set_params(**params_reg[t])
            regression_method_dic[t].fit(self.X, self.y_reg)
        
        self.regression_method_dic = regression_method_dic
        
        return
    
    def predict(self, X_val):
        self.X_val = X_val
        
        y_class_pred = self.classifier_method.predict(self.X_val)    
        #y_reg_pred = np.zeros(len(y_class_pred)).reshape(1,-1)
        y_reg_pred = np.empty(shape=(len(X_val),1))
        print(y_reg_pred.shape)
    
        for p in range(len(y_class_pred)):
            y_reg_pred[p] = self.regression_method_dic[y_class_pred[p]].predict(self.X_val[p].reshape(1,-1))
        
        return y_reg_pred
    
    def optimize_class(self, hp_classifier, values_test, X = None, y = None, fixed_hp = None):
        
        if fixed_hp is None:
            fixed_hp = self.default_params_class
        if X is None:
            X = self.X
        if y is None:
            y = self.y_class
        
        classifier_method_aux_train = self.classifier_method
        classifier_method_aux_test = self.classifier_method
        
        classifier_method_aux_train.set_params(**fixed_hp)
        classifier_method_aux_test.set_params(**fixed_hp)
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=7)
        
        accuracy_train = np.array([])
        accuracy_test = np.array([])
        
        precision_train = np.array([])
        precision_test = np.array([])
        
        print('----------------------------')
        print('Search for the best {} value'.format(hp_classifier))
        print('----------------------------')
        for i in values_test:
            
            classifier_method_aux_train.__setattr__(hp_classifier, i)
            classifier_method_aux_test.__setattr__(hp_classifier, i)
            
            classifier_method_aux_train.fit(X_train, y_train)
            classifier_method_aux_test.fit(X_train, y_train)
            
            y_train_pred = classifier_method_aux_train.predict(X_train)
            y_test_pred = classifier_method_aux_test.predict(X_test)
        
            accuracy_train = np.append(accuracy_train, accuracy_score(y_train, y_train_pred))
            accuracy_test = np.append(accuracy_test, accuracy_score(y_test, y_test_pred))
            
            precision_train = np.append(precision_train, precision_score(y_train, y_train_pred, average='micro'))
            precision_test = np.append(precision_test, precision_score(y_test, y_test_pred, average='micro'))
            
            print('{}: {}, accuracy_train: {},  accuracy_test: {}, precision_train: {}, precision_test: {}'.format(hp_classifier, i, accuracy_train[-1], accuracy_test[-1], precision_train[-1], precision_test[-1]))

        #salvar as metricas em dataframe
    
        result = {'accuracy_train': accuracy_train, 'accuracy_test': accuracy_test,
                  'precicions_train': precision_train, 'precision_test': precision_test}
        result = pd.DataFrame(data=result)
        
        position_max_accuracy = np.where(accuracy_test == max(accuracy_test))
        #position_max_precision = np.where(precision_test == np.max(precision_test))
        best_hp = accuracy_test[position_max_accuracy] 
    
        print('Best {}: {}'.format(hp_classifier,values_test[position_max_accuracy[0][0]]))
        

        plt.figure(dpi=300, figsize=[5,5])
        plt.subplot(121)
        plt.plot(values_test, accuracy_train, label = 'Accuracy train')  
        plt.plot(values_test, accuracy_test, label = '$Accuracy test')
        plt.xticks(values_test)
        plt.yticks(best_hp)
        plt.title('Accuracy')
        plt.xlabel(hp_classifier)
        plt.legend(fontsize=5,loc='best')
        
        plt.subplot(122)
        plt.plot(values_test, precision_train, label = 'Precision train')  
        plt.plot(values_test, precision_test, label = 'Precision test')
        plt.title('Precision')
        plt.xticks(values_test)
        plt.yticks(best_hp)
        plt.xlabel(hp_classifier)  
        plt.legend(fontsize=5, loc='right')
        plt.suptitle('Search Hyper-parameter {}'.format(hp_classifier))
        plt.tight_layout()

        return
    
    
    def optimize_reg(self, regressor_label, hp_regressor, values_test , X, y_reg, y_class, fixed_hp = None, X_val = None, y_reg_val = None):
        
        
        
        if fixed_hp is None:
            fixed_hp = self.default_params_reg
        
        regression_method_from_label = self.regression_method    
        regression_method_aux_train = self.regression_method
        regression_method_aux_test = self.regression_method
        
        regression_method_from_label.set_params(**fixed_hp)
        regression_method_aux_train.set_params(**fixed_hp)
        regression_method_aux_test.set_params(**fixed_hp)
            
        X_dic = {}
        y_class_dic = {}
        y_reg_dic = {}
            
        n_class = len(set(y_class))
        self.n_features = X.shape[1]
            
        for n in range(1, n_class+1):
                
            X_dic[n] = np.empty(shape=(0, self.n_features))
            y_class_dic[n] = np.empty(shape=(0,1))
            y_reg_dic[n] = np.empty(shape=(0,1))
             
        
        
        for i in range(len(y_class)):
            X_dic[y_class[i]] = np.append(X[i],X_dic[y_class[i]]).reshape(-1,self.n_features)
            #y_class_dic[y_class[i]] = np.append(y_class[i], y_class_dic[y_class[i]])
            y_reg_dic[y_class[i]] = np.append(y_reg[i], y_reg_dic[y_class[i]])
            
        if X_val is not None:
            X_dic[regressor_label] = X_val
        if y_reg_val is not None:
            y_reg_dic[regressor_label] = y_reg_val
            #regression_method_train_dic = {}
            #egression_method_test_dic = {}
          
            
        # X_train = {}
        # X_test = {} 
        # y_reg_train = {} 
        # y_reg_test = {}
            
        # for i in range(1, n_class + 1):
            
        X_train, X_test, y_reg_train, y_reg_test = train_test_split(X, y_reg, test_size=0.3, random_state=7)
            
            
            
            #regression_method_train[regressor_label] = regression_method_aux_train
            #regression_method_test_dic[regressor_label] = regression_method_aux_test
        r2_from_label = np.array([])    
        r2_train = np.array([])
        r2_test = np.array([])
          
        mse_from_label = np.array([])
        mse_train = np.array([])
        mse_test = np.array([])
            
        print('----------------------------')
        print('Search for the best {} value'.format(hp_regressor))
        print('----------------------------')
        for i in values_test:
            
            regression_method_from_label.__setattr__(hp_regressor, i)
            regression_method_aux_train.__setattr__(hp_regressor, i)
            regression_method_aux_test.__setattr__(hp_regressor, i)
            
            regression_method_from_label.fit(X_train, y_reg_train)
            regression_method_aux_train.fit(X_train, y_reg_train)
            regression_method_aux_test.fit(X_train, y_reg_train)
            
            y_reg_from_label_pred = regression_method_from_label.predict(X_dic[regressor_label])
            y_reg_train_pred = regression_method_aux_train.predict(X_train)
            y_reg_test_pred = regression_method_aux_test.predict(X_test)
            
            r2_from_label = np.append(r2_from_label, r2(y_reg_dic[regressor_label], y_reg_from_label_pred))
            r2_train = np.append(r2_train, r2(y_reg_train, y_reg_train_pred ))
            r2_test  = np.append(r2_test,  r2(y_reg_test,  y_reg_test_pred))
                
            mse_from_label = np.append(mse_from_label, mse(y_reg_dic[regressor_label], y_reg_from_label_pred))
            mse_train = np.append(mse_train, mse(y_reg_train, y_reg_train_pred ))
            mse_test  = np.append(mse_test,  mse(y_reg_test,  y_reg_test_pred))
                
            print('{}: {}, r2_from_label: {}, mse_from_label: {}'.format(hp_regressor, i, r2_from_label[-1], mse_from_label[-1]))
    
            #salvar as metricas em dataframe
        
        result = {'r2_train': r2_train, 'r2_test': r2_test,'mse_train': mse_train, 'mse_test': mse_test, 'r2_from_label': r2_from_label}
        result = pd.DataFrame(data=result)
            
        position_max_r2 = np.where(r2_from_label == np.max(r2_from_label))
        
            
        print('Best {}: {}'.format(hp_regressor,values_test[position_max_r2[0][0]]))
        
    
        plt.figure(dpi=300)
        plt.subplot(121)
        plt.plot(values_test, r2_train, label = 'train')  
        plt.plot(values_test, r2_test, label = 'test')
        plt.plot(values_test, r2_from_label, label = 'from_label')
        plt.xticks(values_test)
        plt.yticks(r2_test)
        plt.title('$R^2$')
        plt.xlabel(hp_regressor)
        plt.legend(fontsize=5,loc='best')
            
        plt.subplot(122)
        plt.plot(values_test, mse_train, label = 'train')  
        plt.plot(values_test, mse_test, label = 'test')
        plt.plot(values_test, mse_from_label, label = 'from_label')
        plt.title('Mean Squared Error')
        plt.xticks(values_test)
        plt.yticks(mse_test)
        plt.xlabel(hp_regressor)  
        plt.legend(fontsize=5, loc='right')
        plt.suptitle('Search Hyper-parameter {}'.format(hp_regressor))
        plt.tight_layout()
            
                
             #   regression_method_train_dic[t].fit(X_train[t], y_reg_train[t])
            
            #self.regression_method_dic = regression_method_dic
            
        return