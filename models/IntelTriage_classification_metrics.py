#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score, roc_auc_score, auc, classification_report, accuracy_score, precision_recall_fscore_support, precision_recall_curve


# In[3]:


def classification_metrics(y, predicted_probs, prob_threshold = 0.5, print_ConfM = False, plot_PRAUC = False):
    #-----------------------evaluate predictions using x (test or train) and y (test or train) and the predicted probabilities
    #-----------------------returns metrics dataframe
    y_pred_probs = predicted_probs
    y_pred = [0 if prd<=prob_threshold else 1 for prd in predicted_probs]
    print('threshold =', prob_threshold)
    print(confusion_matrix(y, y_pred))
    
    accuracy = accuracy_score(y, y_pred)
    precision, recall, fscore, _ = precision_recall_fscore_support(y, y_pred)
    roc_auc = roc_auc_score(y, predicted_probs)
    # calculate precision and recall for each threshold
    lr_precision, lr_recall, thresholds = precision_recall_curve(y, predicted_probs)
    lr_auc = auc(lr_recall, lr_precision)
    
    if plot_PRAUC:
        plt.figure(figsize=(5, 5))
        plt.title("Precision and Recall Scores as a function of the decision threshold")
        plt.plot(thresholds, lr_precision[:-1], "b--", label="Precision")
        plt.plot(thresholds, lr_recall[:-1], "g-", label="Recall")
        plt.ylabel("Score")
        plt.xlabel("Decision Threshold")
        plt.legend(loc='best')
        plt.show()
        
    metrics_df = pd.DataFrame()
    metrics_df['accuracy'] = [accuracy]
    metrics_df['precision']=[precision[1]]
    metrics_df['specificity']=[recall[0]]
    metrics_df['recall']=[recall[1]]
    metrics_df['AU-ROC']=[roc_auc]
    metrics_df['precision-recall AU']=[lr_auc]
    return metrics_df

