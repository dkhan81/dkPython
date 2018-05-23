# -*- coding: utf-8 -*-
"""
Created on Fri Nov 24 13:51:14 2017

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

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D


sns.set_style('whitegrid')


# Import data
data = pd.read_csv("./0_Data/BreastCancerWisconsin.csv")
print("- Data has {} rows and {} columns.".format(*data.shape))
print("- Column names: ", list(data.columns))


# Split dataset into X and y
X = data.drop(['diagnosis'], axis=1)
X = X.iloc[:, :10]
y = data['diagnosis']


# Standardize data onto unit scale (mean=0 and variance=1)
X = StandardScaler().fit_transform(X)


# Perform PCA
pca = PCA(n_components=None)
Z = pca.fit_transform(X)
print("- Shape of transformed data: ", Z.shape)


# Explained variance ratio of principal components
num_components = pca.n_components_
exp_var = pca.explained_variance_ratio_
cum_exp_var = np.cumsum(exp_var)


# Plot explained variance ratio and cumulative sums
plt.figure(num=1, figsize=(7, 7))
plt.bar(range(num_components), exp_var, alpha=0.5, label='individual explained variance')
plt.step(range(num_components), cum_exp_var, label='cumulative explained variance')
plt.xlabel('Principal components')
plt.ylabel('Explained variance ratio')
plt.legend(loc='best')
plt.show()


# Plot the transformed data (Z) with 2 PCs
plt.figure(num=2, figsize=(7, 7))
for label, color, marker in zip(('B', 'M'), ('blue', 'red'), ('o', '^')):
    plt.scatter(Z[y == label, 0], Z[y == label, 1],
                label=label, color=color, marker=marker, alpha=0.5)
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.legend(loc='best')
plt.tight_layout()
plt.show()


# Plot the transformed data (Z) with 3 PCs
fig = plt.figure(num=3, figsize=(7, 7))
ax = fig.add_subplot(111, projection='3d')
for label, color, marker in zip(('B', 'M'), ('blue', 'red'), ('o', '^')):
    ax.scatter(Z[y == label, 0], Z[y == label, 1], Z[y == label, 2],
               label=label, color=color, marker=marker, alpha=0.5)
ax.set_xlabel('Principal Component 1')
ax.set_ylabel('Principal Component 2')
ax.set_zlabel('Principal Component 3')
ax.legend(loc='best')
plt.show(fig)
