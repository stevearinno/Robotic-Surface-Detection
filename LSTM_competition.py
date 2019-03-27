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
from sklearn.model_selection import cross_val_score

from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense, Dropout, Activation
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from sklearn import preprocessing
from keras.utils import to_categorical
from numpy import argmax
from mlxtend.feature_selection import SequentialFeatureSelector
from keras import regularizers

def main():
    # Load the data, get features, scale data
    foldername = 'data'
    X, X_test_kaggle, y, groups, lenc = load_data(foldername)
    # X, X_test_kaggle = extract_features(X), extract_features(X_test_kaggle)
    # scaler = StandardScaler()
    # X = scaler.fit_transform(X)
    # X_test_kaggle = scaler.transform(X_test_kaggle)
    print(X.shape)
    print(y.shape)
    # print(X[1])

    rs = GroupShuffleSplit(n_splits=100, test_size=0.2)
    for trindex, tsindex in rs.split(X, y, groups):
        X_train, y_train = X[trindex], y[trindex]
        X_test, y_test = X[tsindex], y[tsindex]

    print(X_train.shape)
    inputShape = X_train.shape
    y_train = to_categorical(y_train,9)
    y_test = to_categorical(y_test,9)
    y = to_categorical(y, 9)
    model = Sequential()
    model.add(LSTM(512, return_sequences=True, input_shape=(10, 128), kernel_regularizer=regularizers.l1(0.01)))
    # model.add(LSTM(512, return_sequences=True, input_shape=(10, 128)))
    # model.add(LSTM(512, return_sequences=True, batch_input_shape=(1423,20)))
    model.add(Dropout(0.2))
    model.add(LSTM(64, return_sequences=False, kernel_regularizer=regularizers.l1(0.01)))
    # model.add(LSTM(64, return_sequences=False))
    model.add(Dropout(0.2))
    # model.add(LSTM(32, return_sequences=False, kernel_regularizer=regularizers.l2(0.01)))
    # model.add(Dropout(0.2))

    model.add(Dense(9, activation='softmax', kernel_regularizer=regularizers.l2(0.01)))

    model.compile(loss='categorical_crossentropy',
                  optimizer='sgd',
                  metrics=['accuracy'])

    # score = cross_val_score(model,X,y,cv=cv,scoring='accuracy')
    # print(score)

    model.summary()

    model.fit(X_train, y_train, epochs=20, batch_size=32, validation_data=(X_test, y_test))
    # model.fit(X, y, epochs=20, batch_size=32)

    predictions = model.predict(X_test_kaggle)

    print(predictions[-1])

    predictions_class = [np.argmax(y_pred, axis=None, out=None) for y_pred in predictions]
    # argmax(to_categorical(predictions,9), axis=None, out=None)
    print(predictions_class)
    print(len(predictions_class))
    predictions_class = np.array(predictions_class)
    # print(predictions_class[:, 0])

    # Submission stuff
    labels = lenc.inverse_transform(predictions_class.astype(int))
    with open('results/submission.csv', "w") as fp:
        fp.write("# Id,Surface\n")
        for i in range(len(labels)):
            fp.write("%d,%s\n" % (i, labels[i]))

main()