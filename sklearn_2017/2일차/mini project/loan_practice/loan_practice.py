# Settings =====================================================================
# Load Modules
import os

import pandas as pd
import numpy as np

from imblearn.under_sampling import RandomUnderSampler
from imblearn.under_sampling import TomekLinks
from imblearn.over_sampling import RandomOverSampler
from imblearn.over_sampling import SMOTE
from imblearn.metrics import geometric_mean_score

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

from statsmodels.discrete.discrete_model import Logit
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

import matplotlib.pyplot as plt
import seaborn as sns

# Set Working Directory
os.chdir("D:/Dropbox/Education/2017_SK_Planet/loan_practice")

# Read Data
DF = pd.read_csv("./loan_data.csv")

# Data Exploration =============================================================
# Data Structure
print(DF.shape)
print(DF.dtypes)
print(DF.describe(include = 'all'))

# Correlation Heatmap
sns.set(context="paper", font="monospace")
corr = DF.corr()
plt.figure(figsize=(12, 10))
sns.heatmap(corr, vmin=-1.0, vmax=1.0, square=True,
            cmap=sns.diverging_palette(240, 10, as_cmap=True))

# TODO: 몇몇 변수를 선택하여 pairplot 그려보기




# Bar Chart
sns.countplot(x='TARGET', data=DF)

# TODO: 몇몇 변수를 선택하여 Box Plot을 그리고, TARGET 별로 차이가 나는지 확인



# Data Preprocessing ===========================================================
# Split into X and y
y = DF['TARGET']
X = DF.drop('TARGET', 1)

# TODO: 더미변수 생성하기





# TODO: (Stratified) Train:Validation:Test = 8:1:1로 나누기










# TODO: Train, Validation, Test 데이터셋에 대하여 min-max scaling
scaler = MinMaxScaler()
scaler.fit(X_train)




# Create Datasets --------------------------------------------------------------
# Original Dataset
X_simple, y_simple = X_train.copy(), y_train.copy()

del X_train
del y_train

## UnderSampling
# TODO: Random Undersampling 데이터셋 만들기



# TODO: Tomek Links 데이터셋 만들기



## OverSampling
# TODO: Random Oversampling 데이터셋 만들기



# TODO: SMOTE ('regular') 데이터셋 만들기



## Save Datasets
dataset_names = ['Simple', 'Random Under', 'Tomek', 'Random Over', 'SMOTE']
datasets = [(X_simple, y_simple), (X_rus, y_rus), (X_tl, y_tl),
            (X_ros, y_ros), (X_sm, y_sm)]

# TODO: for, zip을 이용하여 반복적으로 모델을 구축하여 Geometric Mean 계산
# Model Evaluation -------------------------------------------------------------
for       in zip(   ,    ):
    X_train, y_train = 
    
    # Decision Tree
    
    
    
    
    
    print("{} Dataset with Decision Tree: {}".format(ds_name, score))
    
    # Logistic Regression
    
    
    
    
    
    print("{} Dataset with Logistic Regression: {}".format(ds_name, score))
    
    # Random Forest
    
    
    
    
    
    print("{} Dataset with Random Forest: {}".format(ds_name, score))
    
    # k-NN Classifier ----------------------------------------------------------
    neighbors = [1, 3, 5, 7, 9, 11, 13, 15]

    # Validation to choose best k
    scoreList = []
    for k in neighbors:
        
        
        
        
        
    best_index = scoreList.index(max(scoreList))
    best_k = neighbors[best_index]
    
    # Fit and Predict (test data)
    
    
    
    
    
    print("{} Dataset with k-NN (k={}): {}".format(ds_name, best_k, score))


# Check effect of each variable ================================================
# Logistic Regression (Coefficients)
logit = Logit(endog=y_sm, exog=X_sm)
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

