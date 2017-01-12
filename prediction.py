import numpy as np
import platform
from sklearn.ensemble import RandomForestRegressor as RFR
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.pipeline import Pipeline
from feature_engeneering import *

system = platform.system()
    
locations, _ = init()

def training(file):
    X, y = parse_file(file)
    X = np.asarray(X)
    y = np.asarray(y)
    y = y.reshape(len(y),1)
    print('Data stored in memory..')
    mod = model(X, y)
    print('Model learned..')
    return mod

def model(X, y):
    print('Training the model..')
    reg = {}
    for name in locations:
        reg[name] = Pipeline([
            ('rfr', RFR(n_estimators=30, criterion='mse', max_depth=None, min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, 
             max_features='auto', max_leaf_nodes=None, min_impurity_split=1e-07, bootstrap=True, oob_score=False, n_jobs=1, random_state=None, 
             verbose=0, warm_start=False))
            ])   
        
    for i, name in enumerate(locations):
            ind_name = np.where(np.argmax(X[:, -len(locations):], axis=1) == i)[0]
            XX_name = X[ind_name]
            y_name = y[ind_name]
            reg[name].fit(XX_name, y_name)
       
    return reg
    
def predict(X, reg, cross_val=False):
    
    multiplicativeConstant = 1.295
    if not cross_val:
        X = X.reshape(1, len(X))
    y_pred = np.zeros(X.shape[0]).reshape(len(X),1)     
    y_pred_original = np.zeros(X.shape[0]).reshape(len(X),1)  
    
    if cross_val:                               
        for i, name in enumerate(locations):
            ind_name = np.where(np.argmax(X[:, -len(locations):], axis=1) == i)[0]            
            XX_name = X[ind_name].astype(float)  
            y_pred[ind_name] = reg[name].predict(XX_name).reshape(len(XX_name),1)
            y_pred_original[ind_name] = reg[name].predict(XX_name).reshape(len(XX_name),1)
        
        y_pred *= multiplicativeConstant
    
    else:
        i = np.argmax(X[0,-len(locations):])
        name = locations[i]           
        ind_name = np.where(np.argmax(X[:, -len(locations):], axis=1) == i)[0]            
        XX_name = X[ind_name].astype(float)  
        y_pred[ind_name] = reg[name].predict(XX_name).reshape(len(XX_name),1)
        y_pred_original[ind_name] = reg[name].predict(XX_name).reshape(len(XX_name),1)
        
        y_pred *= multiplicativeConstant
         
    return [y_pred, y_pred_original]

def parse_file(file):
    X = []
    y = []
    with open(file, 'r') as data:
        data.readline()
        for line in data:
            row = line.split(';')
            for i in range(len(row)):
                row[i] = float(row[i])

            X.append(row[:len(row)-1]) # all before calls
            y.append(row[len(row)-1]) # calls
    return X, y