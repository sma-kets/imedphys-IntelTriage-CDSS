#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV
from sklearn.metrics import roc_curve, auc, make_scorer, recall_score, accuracy_score, precision_score, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import InputLayer, Conv2D, Dense, Dropout, Flatten, MaxPooling2D
from tensorflow.keras.optimizers import SGD, Adam, RMSprop, Adam, Adamax, Nadam
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier

# In[ ]:
# Function to create model, required for KerasClassifier
def create_model(dropout_rate=0.0, weight_constraint=0,activation='relu', CNN_filters = 4, optimizer='adam',units=32,momentum=0.0, learn_rate=0.001):
    # create model
    model = Sequential()
    model.add(Conv2D(CNN_filters, (2, 2), input_shape=(6,6,1), padding='valid',activation=activation))
    model.add(Conv2D(CNN_filters, (2, 2), padding='valid',activation=activation))
    model.add(Conv2D(CNN_filters, (2, 2), padding='valid',activation=activation))

    model.add(Flatten())
    model.add(Dense(units=32, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    # Compile model
    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    return model

def Random_Search_Hyperparameter_Optimization(X_train1, y_train1, model_name, n_splits, n_iter_search, refit_score): 
    #parameter_list
    parameters2 = {'XGBoost': {
                                'max_depth': range(2, 15, 1),
                                'n_estimators': range(5, 500, 2),
                                'learning_rate': np.arange(0.01,1.0,0.01)}, 
                    'RF':     {'n_estimators': range(5, 500, 2)},
                    'KNN':    {'n_neighbors': range(1,10,1)},
                    'LR':     {'C': [10**-2, 10**-1, 10**0, 10**1, 10**2]},
                    'CNN':    {'batch_size': [4, 6, 8],
                                'epochs': [50,80, 100],
                                'learn_rate': [0.0001, 0.00001, 0.000001],
                                'activation': ['relu', 'tanh', 'sigmoid'],
                                'CNN_filters' : [4,8,16,32]
                              }
                  }

    #define models
    XGBoost = XGBClassifier(n_jobs = -1, use_label_encoder=False, objective = 'binary:logistic', verbose=0) #
    RF = RandomForestClassifier()
    KNN = KNeighborsClassifier()
    LR = LogisticRegression(penalty='l1', solver='liblinear', max_iter = 400)
    CNN = KerasClassifier(build_fn=create_model, verbose=0)

    MODELS = {'XGBoost': XGBoost,'RF': RF, 'KNN': KNN, 'LR': LR, 'CNN': CNN}
    
    #defime metrics
    scoring = {'AUC': 'roc_auc', 
               'Accuracy': make_scorer(accuracy_score), 
               'Precision': make_scorer(precision_score), 
               'Recall': make_scorer(recall_score)
              }
    
    # define K-fold evaluation
    cv = StratifiedKFold(n_splits=n_splits)#, n_repeats=2, random_state=1)
    print(MODELS[model_name])
    #Random Search + K-fold evaluation
    search = RandomizedSearchCV(MODELS[model_name], parameters2[model_name], n_iter=n_iter_search , scoring=scoring, refit=refit_score, 
                                return_train_score=True,cv=cv,verbose=0) #n_jobs=-1 causes the model to use as many available CPU resources as possible.                                                                                                           #It is better to keep this variable to -10 or even lesser in case you are sharing a server with your lab members. More information here. 
    result = search.fit(X_train1, y_train1)
    # summarize result
    print('Best Score: %s' % result.best_score_)
    print('Best Hyperparameters: %s' % result.best_params_)
    return result

