# Settings =====================================================================
# Load Modules
import os

import pandas as pd
import numpy as np

from imblearn.under_sampling import RandomUnderSampler
from imblearn.under_sampling import TomekLinks
from imblearn.over_sampling import SMOTE
from imblearn.metrics import geometric_mean_score

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

from statsmodels.discrete.discrete_model import Logit
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

import matplotlib.pyplot as plt
import seaborn as sns

# Set Working Directory
os.chdir("D:/Dropbox/Education/2017_SK_Planet/p2p_example")

# Read Data
DF = pd.read_csv("./p2p_data.csv")

# Data Exploration =============================================================
# Data Structure
print(DF.shape)
print(DF.dtypes)
print(DF.describe(include='all'))

# Correlation Heatmap
sns.set(context="paper", font="monospace")
corr = DF.corr()
plt.figure(figsize=(10, 8))
sns.heatmap(corr, vmin=-1.0, vmax=1.0, square=True,
            cmap=sns.diverging_palette(240, 10, as_cmap=True))

# Pairplot
select_columns = ['dti', 'tot_hi_cred_lim', 'pct_tl_nvr_dlq', 'TARGET']
sns.pairplot(DF[select_columns], hue='TARGET', size=3, plot_kws={"s": 8})

# Bar Chart
sns.countplot(x='TARGET', data=DF)

# Box Plot
sns.boxplot(y='bc_util', x='TARGET', data=DF, showmeans=True)
sns.boxplot(y='int_rate', x='TARGET', data=DF, showmeans=True)

# Data Preprocessing ===========================================================
# Split into X and y
y = DF['TARGET']
X = DF.drop('TARGET', 1)

# 1-of-C coding (dummy variable)
X = pd.get_dummies(X)
X = X.drop(['verification_status_Not Verified',
            'home_ownership_MORTGAGE',
            'term_ 36 months'], axis=1)
columns = X.columns.tolist()
print(columns)

# (Stratified) Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=0.2,
                                                    random_state=111,
                                                    stratify=y)

# min-max scaling
scaler = MinMaxScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

# Create Datasets --------------------------------------------------------------
# Original Dataset
X_simple, y_simple = X_train.copy(), y_train.copy()

del X_train
del y_train

## UnderSampling
# Random Undersampling
rus = RandomUnderSampler()
X_rus, y_rus= rus.fit_sample(X_simple, y_simple)

# Tomek Links
tl = TomekLinks()
X_tl, y_tl = tl.fit_sample(X_simple, y_simple)

## OverSampling
# SMOTE ('regular')
sm = SMOTE(k_neighbors=5, kind='regular')
X_sm, y_sm = sm.fit_sample(X_simple, y_simple)

## Save Datasets
dataset_names = ['Simple', 'Random Under', 'Tomek', 'Random Over', 'SMOTE']
datasets = [(X_simple, y_simple), (X_rus, y_rus), (X_tl, y_tl), (X_sm, y_sm)]

# Model Construction ===========================================================
# Model Evaluation -------------------------------------------------------------
for ds_name, ds in zip(dataset_names, datasets):
    X_train, y_train = ds
    
    # Decision Tree
    dt = DecisionTreeClassifier()
    dt.fit(X_train, y_train)
    y_pred = dt.predict(X_test)
    score = geometric_mean_score(y_test, y_pred)
    score = round(score, 3)
    print("{} Dataset with Decision Tree: {}".format(ds_name, score))
    
    # Logistic Regression
    logit = Logit(endog=y_train, exog=X_train)
    result = logit.fit(maxiter=1000, disp=0)
    y_pred = logit.predict(params=result.params, exog=X_test).round()
    score = geometric_mean_score(y_test, y_pred)
    score = round(score, 3)
    print("{} Dataset with Logistic Regression: {}".format(ds_name, score))
    
    # Random Forest
    rf = RandomForestClassifier(n_estimators=200)
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)
    score = geometric_mean_score(y_test, y_pred)
    score = round(score, 3)
    print("{} Dataset with Random Forest: {}".format(ds_name, score))
    
# Check effect of each variable ================================================
# Logistic Regression (Coefficients)
logit = Logit(endog=y_rus, exog=X_rus)
result = logit.fit(maxiter=1000, disp=0)
result.summary()

# Get p-values and Coefficients
pval = result.pvalues
coef = result.params

# Get variables with pvalue < 0.05
imp_index = np.where(pval < 0.05)
columns = np.array(columns)

imp_pval = pval[imp_index]
imp_coef = coef[imp_index]
imp_vars = columns[imp_index]

# Plot
plt.figure(figsize=(8,8))
plt.barh(range(len(imp_vars)), imp_coef, color='r')
plt.yticks(range(len(imp_vars)), imp_vars)
plt.show()

# Random Forest (Feature Importances)
rf = RandomForestClassifier(n_estimators=200)
rf.fit(X_rus, y_rus)

importances = rf.feature_importances_
std = np.std([tree.feature_importances_ for tree in rf.estimators_],
             axis=0)
indices = np.argsort(importances)[::-1]

plt.figure(figsize=(8, 8))
plt.title("Feature Importances")
plt.barh(range(X.shape[1]), importances[indices],
         color='r', xerr=std[indices])
plt.yticks(range(X.shape[1]), columns[indices])
plt.show()

