# -*- coding: utf-8 -*-
"""
@author: tut_group_50
"""
import numpy as np
from calculator import running_average

def extract_features(X):
    """
    Extracts the features of the given data array.
    Computes the mean of each channel & adds 10 new features (the std of each 
    channel)
    
    Parameters:
    X - array of data
    
    Returns:
    X - data with 10 additional features
    
    """
    
    temp = X
    
    f3 = []
    for i in range(np.size(X,0)):
        f1 = []
        f2 = []
        for j in range(np.size(X,1)):
            running_avg = running_average(X[i,j], 5)
            org = X[i,j][2:126]
            
            diff = (running_avg - org)
            total_diff = sum(np.abs(diff))
            avg_diff = np.average(np.abs(diff))
            maxmin_diff = np.abs(np.max(org)-np.min(org))
            max_diff = np.max(np.abs(diff))
            
            f1 = np.hstack((total_diff, avg_diff, maxmin_diff, max_diff))
            # 40 pitkä lista featureja, 
            f2 = np.hstack((f2,f1))
        f3.append(f2)
        
        
    temp = np.std(X, axis=2)
    X = np.mean(X, axis=2)
    X = np.concatenate((X, temp), axis=1)
    
    return X
