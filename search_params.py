# -*- coding: utf-8 -*-
"""
finds the best combination of parameters for the functions

@author: Jesse
"""

import numpy as np
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupKFold, GridSearchCV

def search_params(X, y, groups):
    
    group_kfold = GroupKFold(n_splits=2)
    
    randomforest_params = {'n_estimators': np.arange(1000, 8000, 500),
                           'max_depth': np.arange(1, 10),
                           'min_samples_split': np.linspace(0.1, 1.0, 10, endpoint=True),
                           'min_samples_leaf': np.linspace(0.1, 0.5, 5, endpoint=True),
                           'max_features': list(range(1, X.shape[1]))
                           }
    svc_params = {'kernel': ['linear', 'rbf', 'poly'],
                  'gamma': [0.1, 1, 10, 100],
                  'C': [0.1, 1, 10, 100, 1000],
                  'degree': [0, 1, 2, 3, 4, 5, 6]
                  }
    LR_params = {'penalty': ['l1', 'l2'],
                 'C': [0.1, 1, 10, 100, 1000]
                 }
    LR = LogisticRegression()
    SV = SVC()
    RFC = RandomForestClassifier()
    classifiers = [LR, SV, RFC]
    params = [LR_params, svc_params, randomforest_params]
    
    for i in range(len(classifiers)):
        clf = GridSearchCV(classifiers[i], params[i], cv=group_kfold)
        clf.fit(X, y, groups)
        print(clf.cv_results_)
        print(clf.best_params_)
        print(clf.best_score_)
        