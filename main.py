# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
from collections import defaultdict

from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import GroupKFold, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from load_data import load_data
from extract_features import extract_features
from classify import classify, classify_multi

def main():
    
    # Load the data, get features, scale data
    foldername = 'data'
    X, X_test_kaggle, y, groups, lenc = load_data(foldername)
    X, X_test_kaggle = extract_features(X), extract_features(X_test_kaggle)
    print('the shape of X: ' + str(X.shape))
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    X_test_kaggle = scaler.transform(X_test_kaggle)

    classifiers = [RandomForestClassifier(n_estimators=500, max_depth=4)]
    predictions, scores, classifiers, probabilities = classify_multi(classifiers, X, y, X_test_kaggle)
    print(predictions)

    # Submission stuff
    labels = lenc.inverse_transform(predictions[:,0].astype(int)) 
    with open('results/submission.csv', "w") as fp:
        fp.write("# Id,Surface\n")
        for i in range(len(labels)):
            fp.write("%d,%s\n"%(i, labels[i]))
        
main()