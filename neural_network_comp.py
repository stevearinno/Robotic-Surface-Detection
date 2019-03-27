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


import glob
import numpy as np
import os
import matplotlib.pyplot as plt
import tensorflow as tf
from simplelbp import local_binary_pattern
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers.core import Dense, Activation, Flatten
from keras.utils import to_categorical


def main():
    # Load the data, get features, scale data
    foldername = 'data'
    X, X_test_kaggle, y, groups, lenc = load_data(foldername)
    print(X.shape)
    print(y.shape)

    num_featmaps = 32
    num_classes = 2
    # num_epochs = 50
    w, h = 5, 5

    model = Sequential()
    # Layer 1: needs input_shape as well.
    model.add(Conv2D(num_featmaps, (w, h),
                     input_shape=(10,128),
                     activation='relu',
                     padding='same'))
    model.add(MaxPooling2D(pool_size=(4, 4)))

    # # Layer 2:
    model.add(Conv2D(num_featmaps, (w, h),
                     activation='relu',
                     padding='same'))
    model.add(MaxPooling2D(pool_size=(4, 4)))

    model.add(Flatten())
    model.add(Dense(100, activation='softmax'))
    model.add(Dense(2, activation='softmax'))
    model.summary()

    # Question 5
    model.compile(loss='categorical_crossentropy',
                  optimizer='sgd',
                  metrics=['accuracy'])
    # print("X_train shape: " + str(X_train.shape))
    # print("y_train shape: " + str(y_train.shape))
    # print("X_test shape: " + str(X_test.shape))
    # print("y_test shape: " + str(y_test.shape))

    # convert class vectors to binary class matrices
    y_train = to_categorical(y_train, num_classes)
    y_test = to_categorical(y_test, num_classes)

    model.fit(X_train, y_train, epochs=20, batch_size=32, validation_data=(X_test, y_test))

    model.compile(loss='categorical_crossentropy', optimizer='rmsprop')

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    # num_classes = 2
    # # convert class vectors to binary class matrices
    # y_train = to_categorical(y_train, num_classes)
    # y_test = to_categorical(y_test, num_classes)

    model.summary()

    model.fit(X_train, y_train, epochs=20, batch_size=32, validation_data=(X_test, y_test))
    model.compile(loss='categorical_crossentropy',
                  optimizer='sgd',
                  metrics=['accuracy'])



main()