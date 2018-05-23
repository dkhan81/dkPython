# -*- coding: utf-8 -*-
"""
Created on Sat Nov 25 16:30:38 2017

@author: hyungu
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler


X = [[2.5, 0.5, 2.2, 1.9, 3.1, 2.3, 2.0, 1.0, 1.5, 1.1],
     [2.4, 0.7, 2.9, 2.2, 3.0, 2.7, 1.6, 1.1, 1.6, 0.9]]
X = np.asarray(X)

# Plot original data
plt.figure(1, figsize=(9, 9))
plt.scatter(X.T[:, 0], X.T[:, 1], marker='^', s= 100.)
plt.xlim([-1, 4])
plt.ylim([-1, 4])
plt.axhline(y=0, xmin=-1, xmax=4, color='red', linestyle='--')
plt.axvline(x=0, ymin=-1, ymax=4, color='red', linestyle='--')
plt.xlabel('X1', fontsize=25)
plt.ylabel('X2', fontsize=25)
plt.title('original data', fontsize=25)
plt.show()

N = X.shape[1]

# Step 1: Obtain zero-centered data
scaler = StandardScaler(with_mean=True, with_std=False)
X_normalized = scaler.fit_transform(X.T).T
print('- Zero centered X:\n', X_normalized)

# Plot zero-centered data
plt.figure(2, figsize=(9, 9))
plt.scatter(X_normalized.T[:, 0], X_normalized.T[:, 1], marker='^', s= 100.)
plt.xlim([-2, 2])
plt.ylim([-2, 2])
plt.axhline(y=0, xmin=-2, xmax=2, color='red', linestyle='--')
plt.axvline(x=0, ymin=-2, ymax=2, color='red', linestyle='--')
plt.xlabel('X1', fontsize=25)
plt.ylabel('X2', fontsize=25)
plt.title('zero-centered data', fontsize=25)
plt.show()

# Step 2: Calculate the unbiased covariance matrix
S1 = 1 / (N - 1) * np.matmul(X_normalized, X_normalized.T)
S2 = np.cov(X_normalized)
np.testing.assert_almost_equal(S1, S2)
S = S1 = S2

# Step 3: Calculate eigenvalues and eigenvectors (p=2)
eigvals, eigvecs = np.linalg.eig(S)
print("- eigenvalues:\n", eigvals)
print("- eigenvectors:\n", eigvecs)

np.testing.assert_almost_equal(np.matmul(S, eigvecs[:, 1]), eigvals[1] * eigvecs[:, 1])
np.testing.assert_almost_equal(np.matmul(S, eigvecs[:, 0]), eigvals[0] * eigvecs[:, 0])

# Step 5: Get new features
z1 = np.matmul(eigvecs[:, 1], X_normalized)
z2 = np.matmul(eigvecs[:, 0], X_normalized)
Z = np.stack([z1, z2], axis=0)
print("- PC1:\n", z1)
print("- PC2:\n", z2)

# Plot new extracted data
plt.figure(3, figsize=(9, 9))
plt.scatter(Z.T[:, 0], Z.T[:, 1], marker='^', s= 100.)
plt.xlim([-3, 3])
plt.ylim([-3, 3])
plt.axhline(y=0, xmin=-3, xmax=3, color='red', linestyle='--')
plt.axvline(x=0, ymin=-3, ymax=3, color='red', linestyle='--')
plt.xlabel('Z1', fontsize=25)
plt.ylabel('Z2', fontsize=25)
plt.title('Extracted data', fontsize=25)
plt.show()

# Transform and inverse transform with PC1
z1 = np.matmul(eigvecs[:, 1], X_normalized)
X_reconstructed = np.matmul(eigvecs[:, 1].reshape(-1, 1), z1.reshape(1, -1))
