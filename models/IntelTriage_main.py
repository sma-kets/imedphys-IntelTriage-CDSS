#!/usr/bin/env python
# coding: utf-8

# In[1]:


# ------------------------- IMPORTS -------------------------------


# In[2]:


from IntelTriage_data_preprocessing import *
from IntelTriage_feature_selection import *
from IntelTriage_hyperparameter_optimization import *
from IntelTriage_classification_metrics import *

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.ensemble import StackingClassifier

import pandas as pd
import numpy


# In[3]:


# -------------------------- FUNCTIONS ------------------------------


# In[4]:


def get_stacking():
    # define the base models
    level0 = list()
    level0.append(('rf', RandomForestClassifier(**optimization_results['RF'].best_params_, random_state = 7)))
    level0.append(('knn', KNeighborsClassifier(**optimization_results['KNN'].best_params_)))
    level0.append(('lr', LogisticRegression(**optimization_results['LR'].best_params_, penalty='l1', solver='liblinear', max_iter = 400)))
    level0.append(('xgboost', XGBClassifier(**optimization_results['XGBoost'].best_params_, n_jobs = -1, use_label_encoder=False, 
                                               objective = 'binary:logistic', random_state = 7)))
    # define meta learner model
    level1 = LogisticRegression()
    # define the stacking ensemble
    model = StackingClassifier(estimators=level0, final_estimator=level1, cv=10) #, use_probas = True, use_features_in_secondary = True)
    
    return model


# In[5]:


# -------------------------- FILENAMES AND PATHS --------------------


# In[6]:


AHEPA_dataset_path = r'C:\Users\sma_k\my_python_scripts_etc\IntelTriage_\AHEPA_symptoms_unpacked.csv'
public_dataset_path = r'C:\Users\sma_k\my_python_scripts_etc\IntelTriage_\public_dataset.csv'


# In[7]:


# -------------------------- LOAD DATASETS --------------------------


# In[8]:


public_dataset = pd.read_csv(public_dataset_path)
AHEPA_dataset = pd.read_csv(AHEPA_dataset_path) 


# In[9]:


# -------------------------- PREPROCESSING --------------------------


# In[10]:


public_dataset = preprocessing(public_dataset, 'public_dataset', discret_list = [], norm=False)
AHEPA_dataset = preprocessing(AHEPA_dataset, 'AHEPA_dataset', discret_list = [], norm=False)


# In[11]:


# -------------------------- TRAIN - TEST split-----------------------
X_train1, X_test1, y_train1, y_test1= train_test_split(AHEPA_dataset.drop('Αποτέλεσμα',axis=1), AHEPA_dataset['Αποτέλεσμα'], 
                                                    test_size = 0.2, random_state=7, shuffle=False)


# In[12]:


# -------------------------- FEATURE SELECTION -----------------------


# In[ ]:


# 10-fold cross validation
number_of_features_to_select = 36

mean_importance = RF_feature_selection(X_train1, y_train1, rotation=90, max_depth=4, estimators=100, 
                         random_state=7, modeling_type = 'classification', show_plot = True, splits=10).mean(axis=1)

# indices of sorted importances
sorted_ind = np.argsort(mean_importance)[mean_importance.shape[0]-X_train1.shape[0]:mean_importance.shape[0]]

# dimensionality reduction
AHEPA_dataset = AHEPA_dataset[list(X_train1.columns[sorted_ind[-number_of_features_to_select:]]) + ['Αποτέλεσμα']]


# In[16]:


AHEPA_dataset.columns


# In[10]:


# RANDOM SEARCH HYPERPARAMETER OPTIMIZATION


# In[11]:


n_splits = 10
refit_score = 'Accuracy'
MODELS = ['XGBoost', 'RF', 'KNN', 'LR']
n_iter_search = [100, 50, 5, 4]

optimization_results = {}

# tuning
for i, model in enumerate(MODELS):
    optimization_results[model] = Random_Search_Hyperparameter_Optimization(X_train1, y_train1, model, n_splits, n_iter_search[i], refit_score)
    
mean_test_keys = [key for key in optimization_results['XGBoost'].cv_results_.keys() if 'mean_test' in key]
std_test_keys = [key for key in optimization_results['XGBoost'].cv_results_.keys() if 'std_test' in key]
mean_cv_metrics = pd.DataFrame(index = MODELS, columns = [x.split('_')[2] for x in mean_test_keys])
std_cv_metrics = pd.DataFrame(index = MODELS, columns = [x.split('_')[2] for x in std_test_keys])

for model_name in MODELS:
    result = optimization_results[model_name]
    ind_max = np.argmax(result.cv_results_['mean_test_Accuracy'])

    for j, key in enumerate(mean_test_keys):
        mean_cv_metrics.loc[model_name, key.split('_')[2]] = result.cv_results_[key][ind_max]
        std_cv_metrics.loc[model_name, std_test_keys[j].split('_')[2]] = result.cv_results_[std_test_keys[j]][ind_max]


# In[16]:


# TEST GENERALIZATION
# train with best hyperparameters and test on the 20% test dataset that was held out before feature selection

#define models
def pick_model(model_name, best_params):    
    if key == 'XGBoost': model = XGBClassifier(**best_params, n_jobs = -1, use_label_encoder=False, 
                                               objective = 'binary:logistic', random_state = 7) 
    elif key == 'RF':    model = RandomForestClassifier(**best_params, random_state = 7)
    elif key == 'KNN':   model = KNeighborsClassifier(**best_params)
    elif key == 'LR':    model = LogisticRegression(**best_params, penalty='l1', solver='liblinear', max_iter = 400)
    return model

resulted_metrics = pd.DataFrame(index = MODELS, columns = ['accuracy', 'precision', 'specificity', 'recall', 'AU-ROC', 'precision-recall AU'])

for key in MODELS:
    print(f'---------------------------  {key} --------------------------------')
    model = pick_model(key, optimization_results[key].best_params_)

    model.fit(X_train1, y_train1)
    
    if key != 'MLP':  y_pred_probs = model.predict_proba(X_test1)[:, 1]   
    else:             y_pred_probs = model.predict(X_test1)         
    
    print(f'------------------- {key} ------------------')
    resulted_metrics.loc[key] = classification_metrics(y_test1, y_pred_probs, prob_threshold = 0.5, plot_ConfusionM = False).values


# In[40]:


# TEST STACKING utilizing the tuned models
# 10 fold cross validation
splits = 10
stacking_10fold = pd.DataFrame(columns = ['accuracy', 'precision', 'specificity', 'recall', 'AU-ROC', 'precision-recall AU'])
skf = StratifiedKFold(n_splits=splits)

for n, indices in enumerate(skf.split(X_train1, y_train1)):
    
    train_index, test_index = indices
    X_train, X_test = X_train1.iloc[train_index], X_train1.iloc[test_index]
    y_train, y_test = y_train1.iloc[train_index], y_train1.iloc[test_index]
    
    # define and train the model
    model = get_stacking()
    model.fit(X_train, y_train)
    
    # make predictions
    y_pred_probs = model.predict_proba(X_test)[:, 1]   
    y_pred = model.predict(X_test)   
    
    stacking_10fold.loc[str(n)] = classification_metrics(y_test, y_pred_probs, prob_threshold = 0.5, plot_ConfusionM = False).iloc[0].values
    print(stacking_10fold)


# In[43]:


mean_cv_metrics['stacking'] = stacking_10fold.mean()
std_cv_metrics['stacking'] = stacking_10fold.std()
mean_cv_metrics.loc['IGTD + CNN'] = [0.773149, 	0.715519, 	0.728254, 	0.695299]

results.loc['IGTD + CNN'] = [0.7318840579710145, 0.6, 0.8068181818181818, 0.6382978723404256, 0.6185567010309279, 0.7855681818181818, 0.670897428226151]


# In[ ]:


# PLOT TRAINING RESULTS
results.T.plot.bar(colormap = 'winter', rot = 0, alpha=0.6, edgecolor='black', figsize = (10,5))
plt.legend(loc='lower right', bbox_to_anchor=(0.99, 0.04),
          ncol=1, fancybox=True, shadow=True)
plt.title('20% hold-out test dataset')
plt.show()     


# In[ ]:


# PLOT TEST RESULTS

