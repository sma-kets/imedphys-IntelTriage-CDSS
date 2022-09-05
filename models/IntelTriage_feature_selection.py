#!/usr/bin/env python
# coding: utf-8

# In[3]:


from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from matplotlib import pyplot as plt
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
import pandas as pd

# In[2]:


# feature selection basedo on feature importance and cross validation
# input: predictors X_train (dataframe), dependent variable y_train (pandas series), max_depth and estimators for RF, modeling type depending on
# variable y and show_plot->if true feature importance barplot is shown
# output: mean importance rank
def RF_feature_selection(X_train1, y_train1, rotation=90, max_depth=2, estimators=100, 
                         random_state=7, modeling_type = 'regression', show_plot = True, splits=2):
    
    number_of_selected_features = X_train1.shape[1]
    column_names = X_train1.columns

    # BUILD THE MODEL
    if modeling_type == 'regression':
        model = RandomForestRegressor(max_depth=max_depth, n_estimators = estimators, random_state=random_state)
    elif modeling_type == 'classification':
        model = RandomForestClassifier(max_depth=max_depth, n_estimators = estimators, random_state=random_state)

    # stratified k-fold cross validation
    skf = StratifiedKFold(n_splits=splits)

    importances = pd.DataFrame()

    for train_index, test_index in skf.split(X_train1, y_train1):
        X_train, X_test = X_train1.iloc[train_index], X_train1.iloc[test_index]
        y_train, y_test = y_train1.iloc[train_index], y_train1.iloc[test_index]

        # BUILD THE MODEL
        model = RandomForestClassifier(max_depth=max_depth, n_estimators = 4, random_state=7)

        # TRAIN THE MODEL
        model.fit(X_train, y_train)

        # random forest importance
        importance = model.feature_importances_

        importances = pd.concat((importances, pd.Series(importance)), axis=1)

    ind_196 = np.argsort(importances.mean(axis=1))[importances.mean(axis=1).shape[0]-X_train.shape[0]:importances.mean(axis=1).shape[0]]
    mean_importance = importances.mean(axis=1)
    
    if show_plot:
        plt.figure(figsize=(5,15))
        plt.barh([x for x in range(len(mean_importance[ind_196[-number_of_selected_features:]]))], mean_importance[ind_196[-number_of_selected_features:]])
        plt.yticks(range(len(mean_importance[ind_196[-number_of_selected_features:]])), 
                   column_names[ind_196[-number_of_selected_features:]], rotation=0,  fontsize='10', horizontalalignment='right')
        plt.title('RF feature importance')
        plt.show()

    return importances
