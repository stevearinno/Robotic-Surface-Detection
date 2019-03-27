# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
from collections import defaultdict

from sklearn.model_selection import GroupShuffleSplit
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from load_data import load_data
from extract_features import extract_features
from classify import classify


def main():

    foldername = 'data'
    X, X_test_kaggle, y, groups, lenc = load_data(foldername)
    X, X_test_kaggle = extract_features(X), extract_features(X_test_kaggle)

    classifiers = [LinearDiscriminantAnalysis(), 
                   SVC(kernel='linear', probability=True), 
                   SVC(kernel='rbf', probability=True), 
                   LogisticRegression(), 
                   RandomForestClassifier(n_estimators=1000, max_depth=4),
                   KNeighborsClassifier(),
                   ExtraTreesClassifier(n_estimators=1000, max_depth=4)]
    
    names = ['LDA', 'SVC(linear)', 'SVC(rbf)', 'Logistic Regression', 'Random Forest', 'KNeighbors', 'ExtraTrees']
    scores_by_clf = defaultdict(list)
    
    rs = GroupShuffleSplit(n_splits=100, test_size=0.2)
    for trindex, tsindex in rs.split(X, y, groups):
        X_train, y_train = X[trindex, :], y[trindex]
        X_test, y_test = X[tsindex, :], y[tsindex]
        
        print("Training set has classes: ", np.unique(y_train))
        amount = []
        for i in range (len(np.unique(y_train))):
            amount.append(len(np.argwhere(y_train==np.unique(y_train)[i])))
        print(amount)
            
        print("Testing set has classes: ", np.unique(y_test))
        amount = []
        for i in range (len(np.unique(y_test))):
            amount.append(len(np.argwhere(y_test==np.unique(y_test)[i])))
        print(amount)
            
        for i in range(len(classifiers)):
            clf = classifiers[i]
            pred, score, clf, proba = classify(clf, X_train, y_train, X_test, y_test, groups=groups)
            print(np.unique(pred))
            print(names[i], 'score: %.3f \n' % score)
            scores_by_clf[names[i]].append(score)

    for clfname in scores_by_clf.keys():
        print(clfname, np.mean(scores_by_clf[clfname]))
            
        
main()
