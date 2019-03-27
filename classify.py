# -*- coding: utf-8 -*-
"""

@author: tut_group_50
"""
import numpy as np
from sklearn.metrics import accuracy_score

def classify(clf, X_train, y_train, X_test, y_test=None, groups=None):
    
    """
    Params:
    
    clf - classifier to train
    X - training data
    y - training data class labels
    groups - block ids for training data samples
    X_test - testing data
    y_test - testing data class labels
    
    Returns:
    
    pred - the class predictions (for either the split X or X_test)
    score - prediction score for cross-validation - returns 0 if y_test = None
    clf - the trained classifier
    proba - the probability matrix
    """
    
    clf.fit(X_train, y_train)
    pred = clf.predict(X_test)
    proba = clf.predict_proba(X_test)
    
    if (y_test is not None):
        score = accuracy_score(y_test, pred)
    else:
        score = 0

    print('prediction')
    print(pred)
        
    return pred, score, clf, proba
     

def classify_multi(classifiers, X_train, y_train, X_test, y_test=None, groups=None):
    
    """
    Uses the classify-function for multiple classifiers
    
    Returns a matrix of probabilites by the classifiers
    dimensions by amount of (samples, classes, classifiers)
    
    """
    
    num_samples = X_test.shape[0]
    num_classes = 9
    num_classifiers = len(classifiers)
    print(num_samples, num_classes, num_classifiers)
    probabilities = np.zeros((num_samples, num_classes, num_classifiers))
    predictions = np.zeros((num_samples, num_classifiers))
    scores = np.zeros(num_classifiers)
    
    for i in range(num_classifiers):
        pred, score, clf, proba = classify(classifiers[i], X_train, y_train, X_test, y_test, groups)
        predictions[:, i] = pred
        scores[i] = score
        probabilities[:, :, i] = proba
        
    return predictions, scores, classifiers, probabilities
    
          