# -*- coding: utf-8 -*-
"""
@author: tut_group_50
"""

import numpy as np
import csv
from sklearn.preprocessing import LabelEncoder

def load_data(folder):
    """
    Loads the data to numpy arrays 
    
    Parameters:
    folder: foldername in working dir that contains the data or path to folder
    
    Returns:
    X - the provided learning data (np array)
    X_kaggle - the data that needs to be predicted (np array)
    y - labels for the provided learning data (encoded)
    groups - np array with block id for corresponding samples
    lenc - label encoder
    """
    
    
    print("Loading training samples...")
    X = np.load(folder + '/X_train_kaggle.npy')
    
    print("Loading testing samples (Kaggle)...")
    X_test_kaggle = np.load(folder + '/X_test_kaggle.npy')
    y = []
    groups = []
    
    print("Loading labels & groups...")
    with open(folder + '/groups.csv') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        for row in reader:
            if '#' in row[0]:
                continue
            else:
                groups.append(row[1])
                y.append(row[2])
    
    groups = np.array(list(map(int, groups)))
    print(X.shape)
    print(X_test_kaggle.shape)
    print('groups:')
    print(groups)
    print(y)
    print(len(y))
    lenc = LabelEncoder()
    y = lenc.fit_transform(y)
    
    print("Data loading done.")
    return X, X_test_kaggle, y, groups, lenc
