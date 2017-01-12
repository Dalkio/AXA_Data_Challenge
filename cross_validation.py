# -*- coding: utf-8 -*-
"""
Created on Wed Dec  7 22:11:43 2016

@author: guillaume
"""
import numpy as np
from feature_engeneering import *
from prediction import *
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor as RFR
from sklearn.preprocessing import StandardScaler
from datetime import *
from feature_engeneering import *

dico_errors = {'Crises' : 0, 
               'CMS' : 0,
               'Domicile' : 0, 
               'Gestion' : 0, 
               'Gestion - Accueil Telephonique' : 0,
               'Gestion Assurances' : 0, 
               'Gestion Clients' : 0,
               'Gestion DZ' : 0,
               'Gestion Relation Clienteles' : 0,
               'Gestion Renault' : 0,
               'Japon' : 0,
               'Manager' : 0, 
                #'M\xc3\xa9canicien' : 0,
               'M\xe9canicien' : 0,
                #'M\xc3\xa9dical' : 0, 
               'M\xe9dical' : 0, 
               'Nuit' : 0,
               'Prestataires' : 0, 
               'RENAULT' : 0, 
               'RTC' : 0,
               'Regulation Medicale' : 0,
               'SAP' : 0,
               'Services' : 0,
               'Tech. Axa' : 0, 
               'Tech. Inter' : 0,
               'Tech. Total' : 0,
               'T\xc3\xa9l\xc3\xa9phonie' : 0,
               'T\xe9l\xe9phonie' : 0,
               'CAT' : 0
               }

calendar = France()


def Linex(trueY, yPred) :
    alpha = 0.1
    return (np.exp(alpha*(trueY - yPred)) - alpha*(trueY - yPred) - 1)
    
# Loop max computation instead of vectorization in order to avoid MemoryError exception
def maxDiffCustom(a, b, absolute=False, number=1):
    default = 0
    maxArray = default * np.ones(number)
    if len(a) != len(b):
        print("Error dimension comparison")
        return
        
    if absolute == True:
        for i in range(len(a)):
            if np.abs(a[i] - b[i]) > np.min(maxArray):
                minLoc = np.argmin(maxArray)
                maxArray[minLoc] = np.abs(a[i] - b[i])
    else:
        for i in range(len(a)):
            if a[i] - b[i] > np.min(maxArray):
                minLoc = np.argmin(maxArray)
                maxArray[minLoc] = a[i] - b[i]

    while default in maxArray:
        index = np.where(maxArray == default)
        maxArray = np.delete(maxArray, index)
        
    return np.sort(maxArray)[::-1]
        

def crossValidationError(file='data/data_transformed.csv'):
    p_unlabelled = 0.2
    X, y = parse_file(file)   
    print("Done parsing..")
    X = np.asarray(X)
    Y = np.asarray(y)
    Y = Y.reshape(len(Y),1)
    
    # random selection of a portion of X and y to make faster tests
    """sampleSize = 0.1*X.shape[0]
    sampleIdx = np.random.choice(X.shape[0], sampleSize, replace=False)
    X = X[sampleIdx,:]
    Y = Y[sampleIdx,:]"""
    
    X_lab, X_unlab, y_lab, y_unlab = train_test_split(X, Y, test_size=p_unlabelled, random_state=57)
    print("Done splitting labeled/unlabeled ones..")
    
    mod = model(X_lab, y_lab)
    print('Done training..')
    
    yPred, y_pred_original = predict(X_unlab, mod, cross_val=True)
    
    print("Max underestimated error : ")
    print(maxDiffCustom(y_unlab, yPred, False, 10))
    
    linexError = 0 
    sumErrorInCondition = 0
    for i in range(len(y_unlab)):
        linexError += Linex(y_unlab[i], yPred[i])
        location = locations[np.argmax(X_unlab[i,-len(locations):])]
        dico_errors[location] += Linex(y_unlab[i], yPred[i])
        if Linex(y_unlab[i], yPred[i]) > 30:
            sumErrorInCondition += Linex(y_unlab[i], yPred[i])
            date = str(int(X_unlab[i][0])) + '-' +  str(np.argmax(X_unlab[i][1:12])+1) + '-' +  str(int(X_unlab[i][13])) + ' ' + '00:00:00'
            date = datetime.strptime(date, '%Y-%m-%d %H:%M:%S')
            print('--------------------------') 
            print("Error: ", Linex(y_unlab[i], yPred[i]))
            print("True value was: ", y_unlab[i])
            print("Prediction is:  ", yPred[i])
            print("Pred without multiplicator is: ", y_pred_original[i])
            print("Error location: ", locations[np.argmax(X_unlab[i,-len(locations):])])
            print("Percentage of difference between prediction and true y", (y_pred_original[i]-y_unlab[i])/y_pred_original[i])
            print("Date : ", date)
            print('--------------------------')   
    
            print("Total error of samples relevant to condition = ", sumErrorInCondition)        
    print("Error of each call center :")
    for center in dico_errors:
        print(center, " ", str(dico_errors[center]))
    return [linexError/len(y_unlab), linexError, len(y_unlab)]

if __name__ == '__main__':
    meanLinex, linexError ,nbPredictions = crossValidationError()    
    print('Mean error = ' + str(meanLinex), "total error = ", linexError, "number of predictions = ", nbPredictions)
