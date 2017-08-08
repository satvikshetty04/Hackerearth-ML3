'''
Performing Parameter Tuning using Grid Search with Cross Validation
'''

import sys
import datetime
import numpy as np
import pandas as pd
from xgboost.sklearn import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold
from matplotlib import pyplot

# --------------------- Initializing variables

start_time = datetime.datetime.now()
print("Started at: " + str(start_time))
pred_df = pd.DataFrame()

# --------------------- Loading datasets

train = pd.read_csv("Data\\train_pp2.csv", header=0)
test = pd.read_csv("Data\\test_pp2.csv", header=0)


# --------------------- Label Encoding

encode_cols = ['countrycode', 'browserid', 'devid']
for col in encode_cols:
    le = LabelEncoder()
    le.fit(list(train[col].values) + list(test[col].values))
    train[col] = le.transform(list(train[col]))
    test[col] = le.transform(list(test[col]))

train_mod = train[train["click"] == 1]
train = train.append(train_mod, ignore_index=True)
train = train.append(train_mod, ignore_index=True)

# --------------------- Dropping columns
pred_df['ID'] = test['ID']
cols_to_drop = ['ID', 'datetime', 'siteid']
train.drop(cols_to_drop, axis=1, inplace=True)
test.drop(cols_to_drop, axis=1, inplace=True)

# --------------------- Splitting data

Y = train['click']
train.drop('click', axis=1, inplace=True)
X = train


# --------------------- Preparing for Grid Search
#max_depth = 7

learning_rate = [0.05, 0.1, 0.2, 0.3, 0.4]
param_grid = dict(max_depth=[7], learning_rate=learning_rate)
kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=7)
xgbclassifier = XGBClassifier()

grid_search = GridSearchCV(xgbclassifier, param_grid, scoring="neg_log_loss", n_jobs=-1, cv=kfold, verbose=1)
grid_result = grid_search.fit(X, Y)
# summarize results
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
	print("%f (%f) with: %r" % (mean, stdev, param))
# plot
pyplot.errorbar(learning_rate, means, yerr=stds)
pyplot.title("XGBoost max_depth vs Log Loss")
pyplot.xlabel('max_depth')
pyplot.ylabel('Log Loss')
pyplot.savefig('max_depth.png')
