
import numpy as np
import pandas as pd
pd.set_option('display.float_format', lambda x: '%.2f' % x)
pd.set_option('display.max_columns', None)
import matplotlib.pyplot as plt
%config InlineBackend.figure_format = 'retina'
import seaborn as sns; sns.set()
import missingno as msno
import statsmodels.formula.api as sm
from statsmodels.api import add_constant
import matplotlib as mpl
from scipy import stats
import warnings 
warnings.filterwarnings("ignore")
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn import linear_model
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score, median_absolute_error, \
explained_variance_score, confusion_matrix, accuracy_score, precision_score, recall_score
import xgboost as xgb
from sklearn.decomposition import PCA

"""
Pruned Linear Regression (Pruned via P-Value Backward Elimination)
-More usable and interpretable.
-Needs domain-expertise input.
"""

y = df.target_dependent_variable
X = df.drop(['target_dependent_variable_name'], axis = 1) #so everything except the dependent variable
X = add_constant(X)
In [8]:
regressor_OLS = sm.OLS(endog = y, exog = X).fit()

def pass_the_p_s(x, sl):
    selected_columns = []
    corresponding_p_Values = []
    regressor_OLS = sm.OLS(y, x).fit()
    numCols = len(x.columns) 
    for i in range(numCols): 
        if regressor_OLS.pvalues[i] < sl:
            selected_columns.append(x.columns[i])
            corresponding_p_Values.append(regressor_OLS.pvalues[i])
    pass_them_p_s = pd.DataFrame(sorted(list(zip(selected_columns,corresponding_p_Values))\
       ,key = lambda x: abs(x[1]),reverse=False)[:50], columns=['Feature', 'p-Value'])
    return pass_them_p_s

def backwardElimination(x, sl): 
    numCols = len(x.columns) 
    for i in range(numCols): 
        regressor_OLS = sm.OLS(y, x).fit()
        maxVar = max(regressor_OLS.pvalues) 
        if maxVar > sl:
            for j in range(len(regressor_OLS.pvalues)): 
                if (regressor_OLS.pvalues[j] == maxVar): 
                    x = x.drop(x.columns[j], axis = 1) 
    regressor_OLS.summary()
    return x

SL = 0.000001  #most recent one = 0.000001  #old one = 0.00001
X_Pruned = backwardElimination(X, SL)

X_Pruned = add_constant(X_Pruned)
pruned_regressor_OLS = sm.OLS(endog = y, exog = X_Pruned).fit()
pruned_regressor_OLS.summary()