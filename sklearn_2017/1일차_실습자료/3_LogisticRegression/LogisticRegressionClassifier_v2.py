# -*- coding: utf-8 -*-
"""
Created on Thu Nov 23 17:18:30 2017

@author: hyungu
"""

# Import modules
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.metrics import confusion_matrix


sns.set_style('whitegrid')


# Import data
data = pd.read_csv("./0_Data/RedWineQuality.csv")
print("- Data has {} rows and {} columns.".format(*data.shape))
print("- Column names: ", list(data.columns))


# Check class label distribution
fig, ax = plt.subplots(figsize=(5, 5))
ax = sns.countplot(data['quality'], palette='Set1')
plt.tight_layout()
plt.show(fig)


# Split data into X and y; the 'quality' column is the class label
# For simplicity, aggregate classes, so that
# quality 3, 4, 5 corresponds to 'low', and
# quality 6, 7, 8 corresponds to 'high'
X = data.drop(['quality'], axis=1)
y = data['quality'] // 6
y = pd.Series(y)


# Check new class label distribution 
fig, ax = plt.subplots(figsize=(5, 5))
ax = sns.countplot(y, palette='Set2')
plt.tight_layout()
plt.show(fig)


# Split dataset into train (80%) & test (20%) sets
X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=0.2,
                                                    stratify=y,
                                                    random_state=2015010720)


# Standardize train set columnwise, to have zero mean and unit variance
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_train = pd.DataFrame(X_train, columns=X.columns)


# Standardize test data columnwise, by using mean and variance obtained from train set 
X_test = scaler.transform(X_test)
X_test = pd.DataFrame(X_test, columns=X.columns)


# Instantiate a Logistic Regression classifier with arbitrary tree depth
clf_lg = LogisticRegression(C=0.01)
# Fit on training set
clf_lg.fit(X_train, y_train)


# Predict labels of train & test sets
y_train_pred = clf_lg.predict(X_train)
y_test_pred = clf_lg.predict(X_test)
# Show train & test accuracies
print('- Accuracy (Train) : {:.4}'.format(accuracy_score(y_train, y_train_pred)))
print('- Accuracy (Test)  : {:.4}'.format(accuracy_score(y_test, y_test_pred)))
# Show train & test f1 scores
print('- F1 score (Train) : {:.4}'.format(f1_score(y_train, y_train_pred)))
print('- F1 score (Test)  : {:.4}'.format(f1_score(y_test, y_test_pred)))

# Plot ROC Curve
y_test_proba = clf_lg.predict_proba(X_test)
fpr, tpr, _ = roc_curve(y_test, y_test_proba[:, 1])
fig, ax = plt.subplots()
ax.plot([0, 1], [0, 1], 'k--')
ax.plot(fpr, tpr)
ax.set_xlabel('False Positive Rate')
ax.set_ylabel('True positive Rate')
plt.show(fig)
