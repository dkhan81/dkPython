#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 16 12:35:03 2017

@author: hq
"""

import numpy as np
import pandas as pd 

from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from sklearn import tree
from sklearn.metrics import confusion_matrix
from sklearn.datasets import make_classification

from imblearn.under_sampling import RandomUnderSampler
from imblearn.under_sampling import TomekLinks
from imblearn.under_sampling import CondensedNearestNeighbour
from imblearn.under_sampling import OneSidedSelection

from imblearn.over_sampling import RandomOverSampler
from imblearn.over_sampling import SMOTE
from imblearn.over_sampling import ADASYN

train = pd.read_csv("/home/hq/Research/sk_practice/data/creditcard_train.csv")
test = pd.read_csv("/home/hq/Research/sk_practice/data/creditcard_test.csv")

X_train = train.iloc[:,:29]
y_train = train.iloc[:,29]

X_test = test.iloc[:,:29]
y_test = test.iloc[:,29]

tree = tree.DecisionTreeClassifier()
base_tree = tree.fit(X_train, y_train)
baseline = confusion_matrix(y_test, base_tree.predict(X_test))
##############################################################################
### Ranom undersampling 
rus = 
X_resampled, y_resampled, idx_resampled = 

rus_tree = tree.fit(X_resampled, y_resampled)
rus_ = confusion_matrix(y_test, rus_tree.predict(X_test))

##############################################################################
### tomaek Links 
tl = 
X_resampled, y_resampled, idx_resampled = 

tl_tree = tree.fit(X_resampled, y_resampled)
tl_ = confusion_matrix(y_test, tl_tree.predict(X_test))

###############################################################################
### Condensed Nearest Neighbor
cnn = 
X_resampled, y_resampled, idx_resampled = 

cnn_tree = tree.fit(X_resampled, y_resampled)
cnn_ = confusion_matrix(y_test, cnn_tree.predict(X_test))

###############################################################################
### One-side selection
oss = 
X_resampled, y_resampled, idx_resampled = 

oss_tree = tree.fit(X_resampled, y_resampled)
oss_ = confusion_matrix(y_test, oss_tree.predict(X_test))

###############################################################################
### Random oversampling
ros = 
X_resampled, y_resampled = 

ros_tree = tree.fit(X_resampled, y_resampled)
ros_ = confusion_matrix(y_test, ros_tree.predict(X_test))

###############################################################################
### SMOTE ('regular')
sm = 
X_resampled, y_resampled = 

sm_tree = tree.fit(X_resampled, y_resampled)
sm_ = confusion_matrix(y_test, sm_tree.predict(X_test))

###############################################################################
### Borderline-SMOTE
sm_bl = 
X_resampled, y_resampled = 

bl_tree = tree.fit(X_resampled, y_resampled)
bl_ = confusion_matrix(y_test, bl_tree.predict(X_test))

###############################################################################
### Borderline-SMOTE
adas = 
X_resampled, y_resampled = 

adas_tree = tree.fit(X_resampled, y_resampled)
adas_ = confusion_matrix(y_test, adas_tree.predict(X_test))

## result _ baseline 
baseline

## result _ undersampling 
rus_
tl_
cnn_
oss_

## result _ oversampling 
ros_
sm_
bl_
adas_

