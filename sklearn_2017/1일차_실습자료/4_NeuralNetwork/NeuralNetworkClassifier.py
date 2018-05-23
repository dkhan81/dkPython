# -*- coding: utf-8 -*-
"""
Created on Thu Nov 23 15:21:31 2017

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
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix


sns.set_style('whitegrid')


# Import data
data = pd.read_csv("./0_Data/Iris.csv")
print("- Data has {} rows and {} columns.".format(*data.shape))
print("- Column names: ", list(data.columns))


# Visualize a pair plot to examine the bivariate relations of variables
sns.pairplot(data=data, hue='Species', palette='Set1',
             diag_kind='kde',
             diag_kws={'alpha': 0.5},
             plot_kws={'alpha': 0.5})


# Split data into X and y; the 'Species' column is the class label
X = data.drop(['Species'], axis=1)
y = data['Species']


# Split dataset into train (80%) & validation (10%) & test (10%) sets
X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=1/10,
                                                    stratify=y,
                                                    random_state=2015010720)

# Standardize dataset columnwise, to have zero mean and unit variance
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train,
                                                     test_size=1/9,
                                                     stratify=y_train,
                                                     random_state=2015010720)


# Instantiate a Multilayer Perceptron classifier
clf_mlp = MLPClassifier(activation='relu', alpha=1e-4, batch_size=15,
                        hidden_layer_sizes=(10, 5), max_iter=1000,
                        solver='adam', verbose=10, random_state=2015010720)
# Fit on training set
clf_mlp.fit(X_train, y_train)


# Predict labels of train & validation sets
y_train_pred = clf_mlp.predict(X_train)
y_valid_pred = clf_mlp.predict(X_valid)
y_test_pred = clf_mlp.predict(X_test)


# Show train & validation accuracies
print('- Accuracy (Train)       : {:.4}'.format(accuracy_score(y_train, y_train_pred)))
print('- Accuracy (Validation)  : {:.4}'.format(accuracy_score(y_valid, y_valid_pred)))
print('- Accuracy (Test)        : {:.4}'.format(accuracy_score(y_test, y_test_pred)))


# Plot a confusion matrix
cm_test = confusion_matrix(y_test, y_test_pred)
cm_test = pd.DataFrame(cm_test, columns=['Setosa', 'Versicolor', 'Virginica'])
sns.heatmap(data=cm_test, annot=True, annot_kws={'size': 18})
