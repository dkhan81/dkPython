# -*- coding: utf-8 -*-
"""
Created on Thu Nov 16 09:01:24 2017

@author: HQ
"""
import matplotlib.pyplot as plt 
import numpy as np

from sklearn.model_selection import train_test_split
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

def plot_(X_resampled, y_resampled, remove = True):
    # visualization 
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    
    if remove == True:
        idx_samples_removed = np.setdiff1d(np.arange(X_train.shape[0]),
                                       idx_resampled)
    
        idx_class_0 = y_resampled == 0
        plt.scatter(X_resampled[idx_class_0, 0], X_resampled[idx_class_0, 1],
                    alpha=.5, label='Class #0')
        plt.scatter(X_resampled[~idx_class_0, 0], X_resampled[~idx_class_0, 1],
                    alpha=.5, label='Class #1')
        plt.scatter(X_train[idx_samples_removed, 0], X_train[idx_samples_removed, 1],
                    alpha=.5, label='Removed samples')
    
    else:    
        
        idx_class_0 = y_resampled == 0
        plt.scatter(X_resampled[idx_class_0, 0], X_resampled[idx_class_0, 1],
                    alpha=.5, label='Class #0')
        plt.scatter(X_resampled[~idx_class_0, 0], X_resampled[~idx_class_0, 1],
                    alpha=.5, label='Class #1')

    # make nice plotting
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()
    ax.spines['left'].set_position(('outward', 10))
    ax.spines['bottom'].set_position(('outward', 10))
    #ax.set_xlim([-5, 5])
    #ax.set_ylim([-5, 5])
    
    #plt.title('Under-sampling using random under-sampling')
    plt.legend()
    plt.tight_layout()
    plt.show()

def create_dataset(n_samples=1000, weights=(0.5, 0.5), n_classes=2,
                   class_sep=0.8, n_clusters=1):
    return make_classification(n_samples=n_samples, n_features=2,
                               n_informative=2, n_redundant=0, n_repeated=0,
                               n_classes=n_classes,
                               n_clusters_per_class=n_clusters,
                               weights=list(weights),
                               class_sep=class_sep, random_state=0)

#############################################################################
## generate data set (standard normal distribution)

X_syn, y_syn = create_dataset(n_samples = 1000, weights = (0.8, 0.2))
X_train, X_test, y_train, y_test = train_test_split(X_syn,y_syn)

plot_(X_train, y_train, remove = False)

tree = tree.DecisionTreeClassifier()
base_tree = tree.fit(X_train, y_train)
baseline = confusion_matrix(y_test, base_tree.predict(X_test))
##############################################################################
### Ranom undersampling 
rus = RandomUnderSampler(return_indices=True)
X_resampled, y_resampled, idx_resampled = rus.fit_sample(X_train, y_train)

plot_(X_resampled, y_resampled, remove = False)
#plot_(X_resampled, y_resampled, remove = True)

rus_tree = tree.fit(X_resampled, y_resampled)
rus_ = confusion_matrix(y_test, rus_tree.predict(X_test))

##############################################################################
### tomaek Links 
tl = TomekLinks(return_indices=True)
X_resampled, y_resampled, idx_resampled = tl.fit_sample(X_train, y_train)

plot_(X_resampled, y_resampled, remove = False)
plot_(X_resampled, y_resampled, remove = True)

tl_tree = tree.fit(X_resampled, y_resampled)
tl_ = confusion_matrix(y_test, tl_tree.predict(X_test))

###############################################################################
### Condensed Nearest Neighbor
cnn = CondensedNearestNeighbour(return_indices=True)
X_resampled, y_resampled, idx_resampled = cnn.fit_sample(X_train, y_train)

plot_(X_resampled, y_resampled, remove = False)
plot_(X_resampled, y_resampled, remove = True)

cnn_tree = tree.fit(X_resampled, y_resampled)
cnn_ = confusion_matrix(y_test, cnn_tree.predict(X_test))

###############################################################################
### One-side selection
oss = OneSidedSelection(return_indices = True)
X_resampled, y_resampled, idx_resampled = oss.fit_sample(X_train, y_train)

plot_(X_resampled, y_resampled, remove = False)
plot_(X_resampled, y_resampled, remove = True)

oss_tree = tree.fit(X_resampled, y_resampled)
oss_ = confusion_matrix(y_test, oss_tree.predict(X_test))

###############################################################################
### Random oversampling
ros = RandomOverSampler()
X_resampled, y_resampled = ros.fit_sample(X_train, y_train)

plot_(X_resampled, y_resampled, remove = False)

ros_tree = tree.fit(X_resampled, y_resampled)
ros_ = confusion_matrix(y_test, ros_tree.predict(X_test))

###############################################################################
### SMOTE ('regular')

sm = SMOTE(k_neighbors = 5, kind = 'regular')
X_resampled, y_resampled = sm.fit_sample(X_train, y_train)

plot_(X_resampled, y_resampled, remove = False)

sm_tree = tree.fit(X_resampled, y_resampled)
sm_ = confusion_matrix(y_test, sm_tree.predict(X_test))

###############################################################################
### Borderline-SMOTE

sm_bl = SMOTE(k_neighbors = 5, m_neighbors = 2, kind = 'borderline1')
X_resampled, y_resampled = sm_bl.fit_sample(X_train, y_train)

plot_(X_resampled, y_resampled, remove = False)

bl_tree = tree.fit(X_resampled, y_resampled)
bl_ = confusion_matrix(y_test, bl_tree.predict(X_test))

###############################################################################
### Borderline-SMOTE
adas = ADASYN(n_neighbors=3)
X_resampled, y_resampled = adas.fit_sample(X_train, y_train)

plot_(X_resampled, y_resampled, remove = False)

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

