# -*- coding: utf-8 -*-
"""
Created on Thu Nov 23 20:24:10 2017

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
from sklearn.metrics import confusion_matrix
from statsmodels.discrete.discrete_model import Logit


sns.set_style('whitegrid')


# Import data
data = pd.read_csv("./0_Data/BreastCancerWisconsin.csv")
print("- Data has {} rows and {} columns.".format(*data.shape))
print("- Column names: ", list(data.columns))


# Split data into X and y; the 'diagnosis' column is the class label
# Only the first 10 columns will be used
X = data.drop(['diagnosis'], axis=1)
X = X.iloc[:, :10]
y = data['diagnosis']


# Check class label distribution
fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(7, 7))
ax = sns.countplot(y, palette='Set1')
ax.set_title("B : M = {} : {}".format(*y.value_counts()))
plt.tight_layout()
plt.show(fig)


# Check correlation among X variables
fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(15, 15))
ax = sns.heatmap(X.corr(), annot=False, fmt='.1f')
ax.set_title("Correlation Heatmap of X variables")
plt.tight_layout()
plt.show(fig)


# Split dataset into train (90%) & test (10%) sets
X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=0.1,
                                                    stratify=y,
                                                    random_state=2015010720)


# Standardize train set columnwise, to have zero mean and unit variance
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_train = pd.DataFrame(X_train, columns=X.columns)


# Standardize test data columnwise, by using mean and variance obtained from train set 
X_test = scaler.transform(X_test)
X_test = pd.DataFrame(X_test, columns=X.columns)


# Instantiate a Logistic Regression classifier (using scikit-learn)
clf_lg = LogisticRegression()
# Fit on training set
clf_lg.fit(X_train, y_train)


# Predict labels of train & test sets
y_train_pred = clf_lg.predict(X_train)
y_test_pred = clf_lg.predict(X_test)
# Show train & test accuracies
print('- Accuracy (Train) : {:.4}'.format(accuracy_score(y_train, y_train_pred)))
print('- Accuracy (Test)  : {:.4}'.format(accuracy_score(y_test, y_test_pred)))
# Show train & test f1 scores
print('- F1 score (Train) : {:.4}'.format(f1_score(y_train, y_train_pred, pos_label='M')))
print('- F1 score (Test)  : {:.4}'.format(f1_score(y_test, y_test_pred, pos_label='M')))


# Plot a confusion matrix
cm_test = confusion_matrix(y_test, y_test_pred)
cm_test = pd.DataFrame(cm_test, columns=['B', 'M'])
sns.heatmap(data=cm_test, annot=True, annot_kws={'size': 18})


# Logistic regression using 'statsmodels.discrete.discrete_model.Logit'
# 'Logit' expects data in a different format; values of 'y' must be integers 

# Make dataset
y_train_sm = [0 if label == 'B' else 1 for label in y_train]
y_train_sm = pd.Series(y_train_sm)
y_test_sm = [0 if label == 'B' else 1 for label in y_test]
y_test_sm = pd.Series(y_test_sm)


# Instantiate a Logistic Regression classifier (using statsmodels)
logit = Logit(endog=y_train_sm, exog=X_train)
result = logit.fit()
print('- Logistic regression result using statsmodels:\n', result.summary())


# Predict probabilities of train & test sets
y_train_sm_prob = logit.predict(params=result.params, exog=X_train)
y_test_sm_prob = logit.predict(params=result.params, exog=X_test)


# Plot predicted probabilities against train set
plt.figure(1)
plt.scatter(x=range(X_train.shape[0]),
            y=y_train_sm_prob,
            color=['red' if prob >= 0.5 else 'blue' for prob in y_train_sm_prob],
            alpha=0.5)
plt.title('Sigmoid values for train data')
plt.xlabel('observation number')
plt.ylabel('probability')
plt.show()


# Plot predicted probabilities against train set
plt.figure(2)
plt.scatter(x=range(X_test.shape[0]),
            y=y_test_sm_prob,
            color=['red' if prob >= 0.5 else 'blue' for prob in y_test_sm_prob],
            alpha=0.5)
plt.title('Sigmoid values for test data')
plt.xlabel('observation number')
plt.ylabel('probability')
plt.show()


# Convert predicted probabilites to labels (threshold=0.5)
y_train_sm_label = np.rint(y_train_sm_prob)
y_test_sm_label = np.rint(y_test_sm_prob)


# Show train & test accuracies
print('- Accuracy (Train) : {:.4}'.format(accuracy_score(y_train_sm, y_train_sm_label)))
print('- Accuracy (Test)  : {:.4}'.format(accuracy_score(y_test_sm, y_test_sm_label)))
# Show train & test f1 scores
print('- F1 score (Train) : {:.4}'.format(f1_score(y_train_sm, y_train_sm_label)))
print('- F1 score (Test)  : {:.4}'.format(f1_score(y_test_sm, y_test_sm_label)))
